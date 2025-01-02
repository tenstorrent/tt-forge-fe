# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from .base_box3d import BaseInstance3DBoxes
from .utils import rotation_3d_in_axis


class DepthInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in Depth coordinates.

    Coordinates in Depth:

    .. code-block:: none

                    up z    y front (yaw=-0.5*pi)
                       ^   ^
                       |  /
                       | /
                       0 ------> x right (yaw=0)

    The relative coordinate of bottom center in a Depth box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of y.
    Also note that rotation of DepthInstance3DBoxes is counterclockwise,
    which is reverse to the definition of the yaw angle (clockwise).

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicates the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front y           ^
                                 /            |
                                /             |
                  (x0, y1, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
               (x0, y0, z0) + ----------- + --------> right x
                                          (x1, y0, z0)
        """
        # TODO: rotation_3d_in_axis function do not support
        #  empty tensor currently.
        assert len(self.tensor) != 0
        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 0.5, 0)
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=2)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    @property
    def bev(self):
        """torch.Tensor: A n x 5 tensor of 2D BEV box of each box
        in XYWHR format."""
        return self.tensor[:, [0, 1, 3, 4, 6]]


    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or \
        rotation matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns \
                None, otherwise it returns the rotated points and the \
                rotation matrix ``rot_mat_T``.
        """
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, \
            f'invalid rotation angle shape {angle.shape}'

        if angle.numel() == 1:
            rot_sin = torch.sin(angle)
            rot_cos = torch.cos(angle)
            rot_mat_T = self.tensor.new_tensor([[rot_cos, -rot_sin, 0],
                                                [rot_sin, rot_cos, 0],
                                                [0, 0, 1]]).T
        else:
            rot_mat_T = angle.T
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)

        self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T
        if self.with_yaw:
            self.tensor[:, 6] -= angle
        else:
            corners_rot = self.corners @ rot_mat_T
            new_x_size = corners_rot[..., 0].max(
                dim=1, keepdim=True)[0] - corners_rot[..., 0].min(
                    dim=1, keepdim=True)[0]
            new_y_size = corners_rot[..., 1].max(
                dim=1, keepdim=True)[0] - corners_rot[..., 1].min(
                    dim=1, keepdim=True)[0]
            self.tensor[:, 3:5] = torch.cat((new_x_size, new_y_size), dim=-1)

        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                # anti-clockwise
                points.rotate(angle)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(self, bev_direction='horizontal', points=None):
        """Flip the boxes in BEV along given BEV direction.

        In Depth coordinates, it flips x (horizontal) or y (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, None):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction == 'vertical':
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]

        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 0] = -points[:, 0]
                elif bev_direction == 'vertical':
                    points[:, 1] = -points[:, 1]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points




  