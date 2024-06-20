# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import threading
from enum import Enum
from typing import List, Optional, Union, Tuple
import queue

from multiprocessing.synchronize import Event as EventClass
from queue import Queue
import torch
import torch.multiprocessing as mp
from loguru import logger

from .tensor import Tensor, is_equivalent_data_format, pad_pytorch_tensor_to_buda
from .utils import detach_tensors, align_up
from pybuda._C.graph import RuntimeTensorTransform, RuntimeTensorTransformType, Shape
from pybuda._C import DataFormat
from .pybudaglobal import TILE_DIM, create_queue

class TransferType(Enum):
    MP_QUEUE = 1 # read from / write to a queue in shared memory (on host)
    DIRECT = 2   # read/write directly (tilize/untilize)
    NONE = 3     # no explicit transfer (i.e. device will do it on its own), so wrapper does nothing

class DeviceConnector:
    """
    DeviceConnector is a light-weight gasket between two devices, providing mechanism to push/pop data. It
    abstracts the mechanism for pushing and popping out, while implementing data transfer through mp queuees,
    direct tilize/untilize, etc.

    All structures within the class can be pickled and sent to other processes.
    """
    def __init__(self, 
            push_type: TransferType, 
            pop_type: TransferType, 
            shutdown_event: Optional[EventClass],
            queue: Optional[Queue] = None,
            side_queue: Optional[Queue] = None):

        self.push_type = push_type
        self.pop_type = pop_type
        self.shutdown_event = shutdown_event # if the event fires, any blocking actions should stop

        if queue is not None:
            self.queue = queue
        elif self.pop_type == TransferType.MP_QUEUE:
            mp_context = mp.get_context('spawn')
            self.queue = create_queue(mp_context)

        self.side_queue = side_queue

    def shutdown(self):
        pass # children will override

    def initialize(self):
        pass # children will override

    def push_to_side_queue(self, tensors: List[Tensor], clone: bool = False):
        """
        Push to side queue, if one is set, to store debug data
        """
        if self.side_queue is not None:
            if clone:
                tensors = [t.clone() for t in tensors]
            tensors = [t.detach() for t in tensors]
            while True:
                try:
                    self.side_queue.put(tensors) # TODO: timeout and break on shutdown_event
                    return
                except queue.Full as _:
                    if self.shutdown_event is not None and self.shutdown_event.is_set():
                        logger.debug("Aborting side queue put due to shutdown event")
                        return [] # got a signal to shutdown and end the process
                    continue


    def push(self, tensors: List[Tensor]):

        self.push_to_side_queue(tensors)
        if self.push_type == TransferType.MP_QUEUE:
            while True:
                try:
                    self.queue.put(tensors) # TODO: timeout and break on shutdown_event
                    return
                except queue.Full as _:
                    if self.shutdown_event is not None and self.shutdown_event.is_set():
                        logger.debug("Aborting queue put due to shutdown event")
                        return [] # got a signal to shutdown and end the process
                    continue

        
        raise RuntimeError(f"Can't handle push to this type: {type(self)}")

    def read(self) -> List[Tensor]:

        if self.queue is not None:
            while True:
                try:
                    data = self.queue.get(timeout=0.1)
                    return data
                except queue.Empty as _:
                    if self.shutdown_event is not None and self.shutdown_event.is_set():
                        logger.debug("Aborting queue get due to shutdown event")
                        return [] # got a signal to shutdown and end the process
                    continue

        raise RuntimeError("No queue to read from")

    def pop(self):
        if self.queue is not None:
            return # no-op

        raise RuntimeError("Can't handle pop")

    def transfer(self, blocking: bool):
        pass # NOP by default, implemented by some versions


    def empty(self) -> bool:
        if self.queue is None:
            raise RuntimeError("This type of connector can't be polled for emptiness")
        return self.queue.empty()

class DirectPusherDeviceConnector(DeviceConnector):
    """
    Connector in which case one device directly pushes (tilizes) to the other
    """
    def __init__(self, shutdown_event: Optional[EventClass], sequential: bool, pop_type: TransferType = TransferType.NONE, side_queue: Optional[queue.Queue] = None, microbatch=1):
        super().__init__(push_type=TransferType.DIRECT, pop_type=pop_type, shutdown_event=shutdown_event, side_queue=side_queue)
        self.direct_push_queues = None # Will be set after compile
        self.sequential = sequential
        self.tile_broadcast_dims = None
        self.runtime_tensor_transforms : List[RuntimeTensorTransform] = None
        self.constant_tensors = None
        self.microbatch = microbatch
        self.pusher_thread = None

    def pusher_thread_main(self, cmdqueue: queue.Queue):
        logger.info("Pusher thread on {} starting", self)
        while True:
            while True:
                try:
                    cmd = cmdqueue.get(timeout=0.1)
                    break
                except queue.Empty as _:
                    if self.shutdown_event is not None and self.shutdown_event.is_set():
                        logger.debug("Ending pusher thread on {} due to shutdown event", self)
                        return # got a signal to shutdown and end the process
                    continue

            if cmd == "quit":
                return

            logger.debug("Pusher thread pushing tensors")
            self._internal_push(cmd)

    def shutdown(self):
        if self.pusher_thread:
            self.pusher_thread_queue.put("quit")

    def initialize(self):
        # Create threads
        if not self.sequential and not self.pusher_thread:
            self.pusher_thread_queue = queue.Queue(maxsize=3) # don't allow pushes to go too far ahead, or we'll run out of memory
            self.pusher_thread = threading.Thread(target=self.pusher_thread_main, args=(self.pusher_thread_queue,))
            self.pusher_thread.start()

    def _internal_push(self, tensors: List[Tensor]):

        tensor_dtypes = [None] * len(tensors)
        if not self.direct_push_queues:
            print(f"Direct push queues have not been set for {self}")
        assert self.direct_push_queues, "Direct push queues have not been set"
        assert self.tile_broadcast_dims is not None
        assert len(tensors) == len(self.direct_push_queues), (
                f"Incorrect number of tensors provided on input: {len(tensors)} vs {len(self.direct_push_queues)}")
        assert self.runtime_tensor_transforms, "Runtime tensor transforms have not been set"
        assert len(tensors) == len(self.runtime_tensor_transforms)

        self.push_to_side_queue(tensors)

        # Convert to supported tilize conversion format, if needed
        if isinstance(tensors, tuple):
            tensors = list(tensors)

        for i, t in enumerate(tensors):
            if isinstance(t, Tensor):
                tensors[i] = self._convert_tensor_for_tilize(t, self.direct_push_queues[i])
            else:
                tensors[i] = t

        # Handles RuntimeTensorTransform::ReinterpretShape
        for i, t in enumerate(tensors):
            if self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.EmbeddingIndex:
                if isinstance(tensors[i], Tensor):
                    t = t.value()
                assert t is not None
                t = self._embedding_index(t, self.runtime_tensor_transforms[i].original_shape, self.direct_push_queues[i])
                tensors[i] = t
                tensor_dtypes[i] = DataFormat.RawUInt32
            elif self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.ConstantInput:
                assert self.constant_tensors[i] is not None
                tensors[i] = self.constant_tensors[i]
                t = tensors[i]

            if isinstance(tensors[i], torch.Tensor):
                if self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.ReinterpretShape:
                    # TODO: RuntimeTensorTransform could do this transform (for all the RuntimeTensorTransformTypes)
                    t = t.contiguous().view(self.runtime_tensor_transforms[i].reinterpreted_shape.as_list())
                elif self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.Prestride:
                    continue
                tile_r = self.tile_dims[i][0] if self.tile_dims is not None else TILE_DIM
                tile_c = self.tile_dims[i][1] if self.tile_dims is not None else TILE_DIM
                tensors[i] = pad_pytorch_tensor_to_buda(
                    t, self.tile_broadcast_dims[i], squeeze=True, microbatch=self.microbatch, tile_r=tile_r, tile_c=tile_c)
            else:
                reinterpreted_shape = None
                if self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.ReinterpretShape:
                    reinterpreted_shape = self.runtime_tensor_transforms[i].reinterpreted_shape.as_list()
                    tensors[i] = t.to_buda_shape(self.tile_broadcast_dims[i], reinterpret_shape=reinterpreted_shape, clone=False, squeeze=True, microbatch=self.microbatch)
                elif self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.Prestride:
                    pass
                elif self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.NoTransform:
                    tensors[i] = t.to_buda_shape(self.tile_broadcast_dims[i], reinterpret_shape=None, clone=False, squeeze=True, microbatch=self.microbatch)

        # def to_tensor_desc(t: Union[Tensor, torch.Tensor], type: Union[DataFormat, None]) -> PytorchTensorDesc:
        #     if isinstance(t, Tensor):
        #         return t.to_tensor_desc()
        #     return pytorch_tensor_to_tensor_desc(t, df=type)

        # BackendAPI.push_to_queues(self.direct_push_queues, [to_tensor_desc(t, type) for t, type in zip(tensors, tensor_dtypes)], single_input=False)
        self.save_tensors = tensors

    def push(self, tensors: List[Tensor]):

        if not self.sequential:
            self.pusher_thread_queue.put(tensors)
        else:
            self._internal_push(tensors)

class DirectPopperDeviceConnector(DeviceConnector):
    """
    Connector in which case one device produces data directly into queues, and other pops from them
    """
    def __init__(self, shutdown_event: Optional[EventClass], side_queue: Optional[queue.Queue] = None):
        super().__init__(push_type=TransferType.NONE, pop_type=TransferType.DIRECT, shutdown_event=shutdown_event, side_queue=side_queue)
        self.direct_pop_queues = None # Will be set after compile
        self.original_shapes = None
        self.runtime_tensor_transforms = None

    # def read(self) -> List[Tensor]:
    #     assert self.direct_pop_queues is not None, "Direct pop queues have not been set"
    #     if len(self.direct_pop_queues) == 0:
    #         return []
    #     assert self.original_shapes is not None
    #     ret = BackendAPI.read_queues(self.direct_pop_queues, self.original_shapes, self.runtime_tensor_transforms, requires_grad=self.requires_grad, single_output=False, shutdown_event=self.shutdown_event, clone=False)
    #     self.push_to_side_queue(ret)
    #     return ret

    def pop(self):
        assert self.direct_pop_queues is not None, "Direct pop queues have not been set"
        if len(self.direct_pop_queues) == 0:
            return
        # BackendAPI.pop_queues(self.direct_pop_queues, single_output=False)

class DirectPusherPopperDeviceConnector(DirectPusherDeviceConnector):
    """
    Connector between two direct devices (i.e. TT devices)
    """
    def __init__(self, shutdown_event: Optional[EventClass], sequential: bool, side_queue: Optional[queue.Queue] = None):
        super().__init__(pop_type=TransferType.DIRECT, shutdown_event=shutdown_event, sequential=sequential, side_queue=side_queue)
        self.direct_pop_queues = None # Will be set after compile
        self.original_shapes = None
        self.runtime_tensor_transforms = None

    # def read(self) -> List[Tensor]:
    #     assert self.direct_pop_queues is not None, "Direct pop queues have not been set"
    #     if len(self.direct_pop_queues) == 0:
    #         return []
    #     assert self.original_shapes is not None
    #     ret = BackendAPI.read_queues(self.direct_pop_queues, self.original_shapes, self.runtime_tensor_transforms, requires_grad=self.requires_grad, single_output=False, shutdown_event=self.shutdown_event, clone=True)
    #     self.push_to_side_queue(ret)
    #     return ret
        
    def pop(self):
        assert self.direct_pop_queues is not None, "Direct pop queues have not been set"
        if len(self.direct_pop_queues) == 0:
            return
        # BackendAPI.pop_queues(self.direct_pop_queues, single_output=False)


    def transfer(self, blocking: bool):
        """
        Transfer a piece of data from src to dest
        """
        data = self.read()
        self.push(data)


class InputQueueDirectPusherDeviceConnector(DirectPusherDeviceConnector):
    """
    Connector from which we can read, from the given queue, but there are no pushes. This is typically the first
    device in the pipeline.

    It implementes a "transfer" function to transfer 1 set of inputs from the queue into the device.
    """
    def __init__(self, q: Queue, shutdown_event: Optional[EventClass], sequential: bool):
        super().__init__(shutdown_event, sequential)
        self.queue = q

    def transfer(self, blocking: bool):
        """
        Transfer a piece of data from queue to device, if there are any. Optionally block.
        """
        if not blocking and self.queue.empty():
            return 

        data = self.read()
        self.push(data)

class OutputQueueDirectPoppperDeviceConnector(DirectPopperDeviceConnector):
    """
    Connector that has an external queue that pushes go to. No reading through this connector is allowed.

    It implementes a "transfer" function to transfer 1 set of outputs from device to the queue
    """
    def __init__(self, q: Queue, shutdown_event: Optional[EventClass], side_queue: Optional[queue.Queue] = None):
        super().__init__(shutdown_event, side_queue=side_queue)
        self.queue = q

    def transfer(self, blocking: bool):
        """
        Transfer a piece of data from device to read queue. Optionally blocking.
        """
        if not blocking:
            raise NotImplementedError("Non-blocking transfer on output not implemented yet")

        data = self.read()
        self.queue.put([t.clone().detach() for t in data])  # Need to clone, otherwise popping will erase the tensor
        self.pop()
