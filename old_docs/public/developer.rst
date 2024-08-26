Developer Reference
===================

Low Level Kernels
-----------------

Overview
********

SFPI is the programming interface to the SFPU.  It consists of a C++ wrapper
around a RISCV GCC compiler base which has been extended with vector data types and
__builtin intrinsics to generate SFPU instructions.  SFPI is supported on Grayskull
and Wormhole version B0.

The primary design goal of the wrapper is for all but the SFPU instructions to
compile to nothing so as to incur zero runtime overhead. Another goal is to,
over time, push functionality out of the wrapper and into the compiler.

Compiler Options/Flags
**********************

The following flags must be specified to compile SFPI kernels:

.. code-block:: c++

  -m<arch> -fno-exceptions

where ``arch`` is one of:

  * grayskull
  * wormhole

Note that the arch specification above overrides any ``-march=<xyz>`` to either
``-march=rv32iy`` for grayskull or ``-march=rv32iw`` for wormhole.

Further, the following options disable parts of the SFPI enabled compiler:

  * ``-fno-rvtt-sfpu-warn``: disable sfpu specific warnings/errors
  * ``-fno-rvtt-sfpu-combine``: disable sfpu instruction combining
  * ``-fno-rvtt-sfpu-cc``: disable sfpu CC optimizations
  * ``-fno-rvtt-sfpu-replay``: disable sfpu REPLAY optimizations (wormhole only)

Example
*******

Before going into details, below is a simple example of SFPI code:

.. code-block:: c++

    void silly(bool take_abs)
    {
        // dst_reg[n] loads into a temporary LREG
        vFloat a = dst_reg[0] + 2.0F;

        // This emits a load, move, mad (on GS uses the "+/0 .5" feature of MAD)
        dst_reg[3] = a * -dst_reg[1] + vConst0p6929 + 0.5F;

        // This emits a load, loadi, mad (a * dst_reg[] goes down the mad path)
        dst_reg[4] = a * dst_reg[1] + 1.2F;

        // This emits two loadis and a mad
        dst_reg[4] = a * 1.5F + 1.2F;

        // This emits a loadi (into tmp), loadi (as a temp for 1.2F) and a mad
        vFloat tmp = s2vFloat16a(value);
        dst_reg[5] = a * tmp + 1.2F;
    
        v_if ((a >= 4.0F && a < 8.0F) || (a >= 12.0F && a < 16.0F)) {
            vInt b = exexp_nodebias(a);
            b &= 0xAA;
            v_if (b >= 130) {
                dst_reg[6] = setexp(a, 127);
            }
            v_endif;
        } v_elseif (a == s2vFloat16a(3.0F) {
            // RISCV branch
            if (take_abs) { 
                dst_reg[7] = abs(a);
            } else {
                dst_reg[7] = a;
            }
        } v_else {
            vInt exp = lz(a) - 19;
            exp = ~exp;
            dst_reg[8] = -setexp(a, exp);
        }
        v_endif;
    }

The main things to note from the example are:

  * Constants are expressed as scalars but are expanded to the width of the vector
  * ``v_if`` (and related) predicate execution of vector operations such that only enabled vector elements are written
  * The compiler views ``v_if`` and ``v_elseif`` as straight-line code, ie, both sides of the conditionals are executed
  * RISCV conditional and looping instructions work as expected (only one side executed)
  * Math expressions for vectors work across all enabled vector elements
  * Presently, ``v_endif`` is required to close out all ``v_if``/``v_elseif``/``v_else`` chains

Details
*******

Namespace
+++++++++

All the data types/objects/etc. listed below fall within the ``sfpi``
namespace.

User Visible Data Types
+++++++++++++++++++++++

The following data types are visible to the programmer:

  * ``vFloat``
  * ``vInt``
  * ``vUInt``
  * enum ``LRegs``

Each of the ``v`` types is a strongly typed wrapper around the weakly typed compiler
data type ``__rvtt_vec_t``.  On Grayskull this is a vector of 64 19 bit values while on Wormhole this is a vector of 32 32 bit values.

LRegs are the SFPU's general purpose vector registers.  ``LRegs`` enumerates
these registers.

User Visible Constants
++++++++++++++++++++++

Constant registers are implemented as objects which can be referenced
whereever a vector can be used.

  * Grayskull:

    * ``vConst0``
    * ``vConst0p6929``
    * ``vConstNeg1p0068``
    * ``vConst1p4424``
    * ``vConst0p8369``
    * ``vConstNeg0p5``
    * ``vConst1``
    * ``vConstNeg1``
    * ``vConst0p0020``
    * ``vConstNeg0p6748``
    * ``vConstNeg0p3447``
    * ``vConstTileId``, enumerates the vector elements: [0..63]

* Wormhole:

  * ``vConst0``
  * ``vConst1``
  * ``vConst0p8373``
  * ``vConstNeg1``
  * ``vConstTileId``, counts by two through the vector elements: [0, 2, 4..62]
  * ``vConstFloatPrgm0``, ``vConstIntPrgm0``
  * ``vConstFloatPrgm1``, ``vConstIntPrgm1``
  * ``vConstFloatPrgm2``, ``vConstIntPrgm2``

User Visible Objects
++++++++++++++++++++

 * ``dst_reg[]`` is an array used to access the destination register
 * ``l_reg[]`` is an array used to load/store to specific SFPU registers

Macros
++++++

The only macros used within the wrapper implement the predicated conditional
processing mechanism.  These (of course) do not fall within the SFPI namespace
and for brevity run some chance of a namespace collision.  They are:

  * ``v_if()``
  * ``v_elseif()``
  * ``v_else``
  * ``v_endif``
  * ``v_block``
  * ``v_endblock``
  * ``v_and()``

The conditionals work mostly as expected but note the required ``v_endif`` at
the end of an if/else chain.  Forgetting this results in compilation
errors as the ``v_if`` macro contains a ``{`` which is matched by the ``v_endif``.

``v_block`` and ``v_and`` allow for the following code to progressively "narrow" the CC
state:

.. code-block:: c++

    v_block {
        for (int x = 0; x < n; x++) {
            v1 = v1 - 1;
            v_and (v1 >= 0);
            v2 *= 2;
        }
    }
    v_endblock;

``v_and`` can be used inside any predicated conditional block (i.e., a ``v_block``
or a ``v_if``).

Data Type Details
+++++++++++++++++

vFloat
^^^^^^

  * Assignment: from float, dst_reg[n]
  * Conversion: ``reinterpret<AnotherVecType>()`` converts, in place, between vInt and vUInt and vFloat
  * Immediate loads: see section :ref:`Immediate Floating Point Values` below
  * Operators: ``+``/``-``/``*`` should work as expected with dst_reg[n], vFloat and vConst
  * Conditionals: All 6 (``<``, ``<=``, ``==``, ``!=``, ``>=``, ``>``) are supported.  Note that ``<=`` and ``>`` pay a performance penalty relative to the others

vInt
^^^^

  * Assignment: from integer, dst_reg[n]
  * Conversion: ``reinterpret<AnotherVecType>()`` converts, in place, between vFloat and vUInt
  * Operators: ``&``, ``&=``, ``|``, ``|=``, ``~``, ``^``, ``^=``, ``<<`` and ``+``, ``-``, ``+=``, ``-=``, ``++``, ``--``.  (there is no signed right shift on Grayskull or Wormhole)
  * Conditionals: All 6 (``<``, ``<=``, ``==``, ``!=``, ``>=``, ``>``) are supported.  Note that ``<=`` and ``>`` pay a performance penalty relative to the others

vUInt
^^^^^

  * Assignment: from unsigned integer, dst_reg[n]
  * Conversion: ``reinterpret<AnotherVecType>()`` converts, in place, between vFloat and vInt
  * Operators: ``&``, ``&=``, ``|``, ``|=``, ``~``, ``^``, ``^=``, ``<<``, ``>>`` and ``+``, ``-``, ``+=``, ``-=``, ``++``, ``--``
  * Conditionals: All 6 (``<``, ``<=``, ``==``, ``!=``, ``>=``, ``>``) are supported.  Note that ``<=`` and ``>`` pay a performance penalty relative to the others

Note that on Wormhole, the destination register format is always determined by the run time.  So, for example, reading a vInt when the format is set to float32 gives unexpected results.

Library
+++++++

Below ``Vec`` means any vector type.

Grayskulll and Wormhole
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

    vInt exexp(const vFloat v)
    vInt exexp_nodebias(const vFloat v)

    vInt exman8(const vFloat v)
    vInt exman9(const vFloat v)

    vFloat setexp(const vFloat v, const uint32_t exp)
    vFloat setexp(const vFloat v, const Vec[U]Short exp)

    vFloat setman(const vFloat v, const uint32_t man)
    vFloat setman(const vFloat v, const Vec[U]Short man) // Grayskull HW bug, is this useful?

    vFloat setsgn(const vFloat v, const int32_t sgn)
    vFloat setsgn(const vFloat v, const vFloat sgn)
    vFloat setsgn(const vFloat v, const vInt sgn)

    vFloat addexp(const vFloat v, const int32_t exp)

    vFloat lut(const vFloat v, const vUInt l0, const vUInt l1, const vUInt l2, const int offset)
    vFloat lut_sign(const vFloat v, const vUInt l0, const vUInt l1, const vUInt l2, const int offset)

    vInt lz(Vec v)

    vFloat abs(vFloat v)
    vInt abs(vInt v)

    vUInt shft(const vUInt v, const vInt amt) // positive amt shifts left, negative shifts right

Wormhole only
^^^^^^^^^^^^^

.. code-block:: c++

    void vec_swap(Vec, Vec)
    void vec_min_max(Vec, Vec)

    Vec subvec_shflror1(Vec v)
    Vec subvec_shflshr1(Vec v)

    void subvec_transp(Vec, Vec, Vec, Vec)

    vInt lz_nosgn(Vec v)

    vFloat int_to_float(vInt in, int round_mode = 1)                 // 0=even, 1=stochrnd
    vUInt float_to_fp16a(vFloat in, int round_mode = 1)
    vUInt float_to_fp16b(vFloat in, int round_mode = 1)
    vUInt float_to_uint8(vFloat in, int round_mode = 1)
    vUInt float_to_int8(vFloat in, int round_mode = 1)
    vUInt int32_to_uint8(vInt in, vUInt descale, int round_mode = 1)
    vUInt int32_to_uint8(vInt in, unsigned int descale, int round_mode = 1)
    vUInt int32_to_int8(vInt in, vUInt descale, int round_mode = 1)
    vUInt int32_to_int8(vInt in, unsigned int descale, int round_mode = 1)
    vUInt float_to_uint16(vFloat in, int round_mode = 1)
    vUInt float_to_int16(vFloat in, int round_mode = 1)
    
Immediate Floating Point Values
+++++++++++++++++++++++++++++++

Assigning a float to a vFloat behaves slightly different on Grayskull vs Wormhole. 
On Grayskull, the value is interpreted as an fp16b; use the conversion routines below
to explicitly specify the format.  On Wormhole, the floating point value is converted
to an fp16a, fp16b, or fp32 by first looking to see if the range fits in fp16b
and if not using fp16a (or fp32).  If the value is not known at compile time,
then it is loaded as an fp32.  Note that on Wormhole fp32 loads take 2 cycles.

For more explicit conversions, use one of the classes ``s2vFloat16a`` and
``s2vFloat16b``.  Each takes either an integer or floating point value.  Floating
point immediate values are converted at compilation time and incur no overhead.
Floating point variables that are not known at compilation time are converted at run
time.  An integer value loaded into floating point vector (via one of the
conversion routines) is treated as a bit pattern and incurs no overhead, see
examples below.

Note: fp16a conversions do not presently handle denorms/nans, etc. properly.

Example uses:

.. code-block:: c++

    vFloat x = 1.0f;               // Load fb16b value
    vFloat x = 500000.0f;          // GS load fp16b value, WH fp32 value
    vFloat x = s2vFloat16a(3.0F);  // Load fp16a value, no overhead
    unsigned int ui = 0x3c00;
    vFloat x = s2vFloat16a(ui);    // Load fp16a value (1.0F), no overhead
    float f = 1.0F;
    vFloat x = s2vFloat16a(f);     // Load fp16a value, overhead if value cannot be determined at compile time

Boolean Operators
+++++++++++++++++

All conditionals operating on base types can be combined with any of ``&&``, ``||``, ``!``.

vBool
+++++

``vBool`` doesn't exist yet, but the functionality can be obtained by executing
conditional instructions outside of a ``v_if`` and assigning the result to a
``vInt``.  This can be useful to, e.g., use RISCV code to conditionally generate
an SFPU predicate.  For example, the following function evaluates different
predicated conditionals based on the value of a function parameter:

.. code-block:: c++

    sfpi_inline vInt sfpu_is_fp16_zero(const vFloat& v, uint exponent_size_8)
    {
        if (exponent_size_8) {
            return v == 0.0F;
        } else {
            vInt tmp = 0x3800; // loads {0, 8'd112, 10'b0}
            tmp += reinterpret<vInt>(v);
            return tmp == 0;
        }
    }

which may be called by:

.. code-block:: c++

    v_if (sfpu_is_fp16_zero(v, exponent_size_8)) {
        ...
    }
    v_endif;

If exponent_size_8 is known at compile time, this has no overhead.  If not,
the predication is determined at runtime.

Assigning and Using Constant Registers
++++++++++++++++++++++++++++++++++++++

Programmable constant registers (Wormhole only) are accessed and assigned just
like any other variables, for example:

.. code-block:: c++

    vConstFloatPrgm0 = 3.14159265;
    vFloat two_pi = 2.0f * vConstFloatPrgm0;

Writing to a constant register first loads the constant into a temporary LReg
then assigns the LReg to the constant register and so takes 1 cycle longer
than just loading an LReg.  Accessing a constant register is just as fast as
accessing an LReg.  Loading a constant register loads the same value into all
vector elements.

Assigning LRegs
+++++++++++++++

Some highly optimized code may call a function prior to the kernel to
pre-load values into specific LRegs and then access those values in
the kernel.  Note that if the register's value must be preserved when
the kernel exits, you must restore the value explicitly by assigning
back into the LReg.

For example:

.. code-block:: c++

    vFloat x = l_reg[LRegs::LReg1];  // x is now LReg1
    vFloat y = x + 2.0f;
    l_reg[LRegs::LReg1] = x;         // this is necessary at the end of the function
                                     // to preserve the value in LReg1 (if desired)

Miscelaneous
************

Register Pressure Management
++++++++++++++++++++++++++++

Note that the wrapper introduces temporaries in a number of places.  For
example:

.. code-block:: c++

  dst_reg[0] = dst_reg[0] + dst_reg[1];

loads dst_reg[0] and dst_reg[1] into temporary LREGs (as expected).

The compiler will not spill registers.  Exceeding the number of registers
available will result in the cryptic: ``error: cannot store SFPU register
(reigster spill?) - exiting!`` without a line number.

The compiler does a reasonable job with lifetime analysis when assigning
variables to registers.  Reloading or recalculating results helps the compiler
free up and re-use registers and is a good way to correct a spilling error.

Grayskull has 4 general purpose LRegs, Wormhole has 8.

Optimizer
+++++++++

There is a basic optimizer in place.  The optimization philosophy to date is to enable the programmer
to write optimal code.  This is different from mainstream compilers which may generate optimal code
given non-optimal source.  For example, common sub-expression elimination and the like are not
implemented.  The optimizer will handle the following items:

  * MAD generation (from MUL/ADD)
  * MULI, ADDI generation (from MUL + const, or ADD + const)
  * Adding a 0.5f to the end of ADD/MULL/MAD/MULI/ADDI (Grayskull only)
  * Swapping the order of arguments to instructions that use the destination-as-source, e.g., SFPOR to minimize the need for register moves
  * CC enables (PUSHC, POPC, etc.)
  * Instruction combining for comparison operations.  For example, a subtract of 5 followed by a compare against 0 gets combined into one operation
  * Wormhole only: NOP insertion for instructions which must be followed by an independent instruction or NOP.  Note that this pass (presently) does not move instructions to fill the slot but will skip adding a NOP if the next instruction is independent.  In other words, reordering your code to reduce dependent chains of instructions may improve performance

There is a potential pitfall in the above in that the MAD generator could
change code which would not run out of registers with, say, a MULI followed by
an ADDI into code that runs out of registers with a MAD.  (future todo to fix this).

SFPREPLAY
+++++++++

The ``SFPREPLAY`` instruction available on Wormhole allows the RISCV processor
to submit up to 32 SFP instructions at once.  The compiler looks for sequences
of instructions that repeat, stores these and then "replays" them later.

The current implemention of this is very much first cut: it does not handle
kernels with rolled up loops very well.  Best performance is typically attained by
unrolling the top level loop and then letting the compiler find the repetitions
and replace them with ``SFPREPLAY``.  This works well when the main loop
contains < 32 instructions, but performance starts to degrade again as the
number of instructions grows (future work).

The other issue that can arise with ``SFPREPLAY`` is that sometimes the last
unrolled loop of instructions uses different registers than the prior
loops resulting in imperfect utilization of the replay.


Emulation
+++++++++

There is an emulator for the SFPU that works at the __builtin level.
Compilation and runtime are extremely fast (sub 1 second) so this may be
useful during development.

Look in the file main.cc in the ``sfpi`` submodule under ``src/ckernels``, there
is an example kernel there to lead the way.

The main difference between compilation and running on HW is that the emulator
has an infinite number of registers and so code that runs there may fail on
the HW due to spilling.  The ``Makefile`` builds for both rv32 (generating a
``.S`` file) and x86 (to run through emulation) and so an "out of registers"
message for rv32 tells you you have work to do.

The emulator for WH is not fully implemented (missing some of the new WH specific instructions)

Tools
+++++

The sfpi submodule contains a ``tools`` directory.  ``cd`` into that directory and
type ``make`` to build ``fp16c`` which is a converter that converts floating point
values to fp16a, fp16b and the LUT instruction's fp8 as well as the other way
(integer to float/fp16a/fp16b/fp8).  This is useful for writing optimal code or
looking through assembly dumps.

Pitfalls/Oddities
*****************

Register Spilling
+++++++++++++++++

The compiler does not implement register spilling.  Since Grayskull only has 4
LRegs, running out of registers is a common occurence.  If you see the
following: ``error: cannot store SFPU register (reigster spill?) - exiting!``
you have most likely run out of registers.

Error Messages
++++++++++++++

Unfortunately, many errors are attributed to the code in the wrapper rather than in the code
being written.  For example, using an unitialized variable would show an error at a macro
called by a wrapper function before showing the line number in the user's code.

Function Calls
++++++++++++++

There is no abi and none of the vector types can be passed on the stack.
Therefore, all function calls must be inlined.  To ensure this use
``sfpi_inline``, which is defined to ``__attribute__((always_inline))`` on GCC.

Unnecessary Moves
+++++++++++++++++

The gcc compiler occasionally moves a value from one register to another
for no apparent reason.  At this point it appears there is nothing that can
be done about this besides hoping that the issue is fixed in a future version
of gcc.

Limitations
+++++++++++

  * Forgetting a ``v_endif`` results in mismatched {} error which can be confusing (however, catches the case where a ``v_endif`` is missing!)
  * In general, incorrect use of vector operations (e.g., passing a scalar instead of a vector) result in warnings/errors within the wrapper rather than in the calling code
  * Keeping too many variables alive at once (4 on GS) requires register spilling which is not implemented and causes a compiler abort

Limitations
