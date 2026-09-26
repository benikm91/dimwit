package dimwit.tensor

/** Padding options for convolution operations.
  * SAME: Output size is the same as input size (with appropriate padding).
  * VALID: No padding, output size is reduced based on kernel size.
  *
  * Refer to JAX documentation for more details on padding behavior.
  * https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html
  */
enum Padding:
  case SAME, VALID

/** Stride of a 1D convolution. */
type Stride1[S1] = AxisExtent[S1]

/** Stride of a 2D convolution. */
type Stride2[S1, S2] = (AxisExtent[S1], AxisExtent[S2])

/** Stride of a 3D convolution. */
type Stride3[S1, S2, S3] = (AxisExtent[S1], AxisExtent[S2], AxisExtent[S3])
