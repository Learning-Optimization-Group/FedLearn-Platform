package com.fedlearn.mobile.datasets

/** What one example of a run's data must be: a float vector, or an 8-bit image and how the contract prepares it. */
sealed interface DataShape {
    /** Examples per record, once prepared. */
    val elementCount: Int

    data class Vector(val width: Int) : DataShape {
        override val elementCount get() = width
    }

    /**
     * An image of [height] x [width] x [channels] 8-bit values (HWC), prepared by the contract's ImageToUnitTensor
     * and, when [mean] and [std] are given, NormalizeChannels. The snapshot's inputs are the prepared CHW float32.
     */
    data class Image(
        val height: Int,
        val width: Int,
        val channels: Int,
        val mean: List<Float>?,
        val std: List<Float>?,
    ) : DataShape {
        init {
            require(height in 1..MAX_SIDE && width in 1..MAX_SIDE && channels in listOf(1, 3))
            require((mean == null) == (std == null))
            require(mean == null || (mean.size == channels && std!!.size == channels))
        }

        override val elementCount get() = height * width * channels

        /**
         * How the values were prepared, recorded in the snapshot so a run with other preparation never trains it:
         * `image:HxWxC`, then `;mean=..;std=..` as float32 bit patterns in hex. The app builds the same string from a
         * run's contract.
         */
        val transformsId: String
            get() = "image:${height}x${width}x$channels" + if (mean == null) "" else
                ";mean=${mean.joinToString(",") { bits(it) }};std=${std!!.joinToString(",") { bits(it) }}"

        private fun bits(v: Float) = "%08x".format(java.lang.Float.floatToRawIntBits(v))

        companion object {
            const val MAX_SIDE = 4096
        }
    }
}

/**
 * The execution contract's image transforms: element (c, y, x) of the CHW result is `p / 255` for the pixel value p
 * at (y, x, c), then `(v - mean[c]) / std[c]`, each one binary32 operation rounded to nearest-even, as the contract
 * states. Kotlin Float arithmetic is binary32, so this matches torchvision's ToTensor and Normalize to the bit.
 */
object ImageTransforms {
    fun toTensor(pixels: ByteArray, shape: DataShape.Image): FloatArray {
        val (h, w, c) = Triple(shape.height, shape.width, shape.channels)
        require(pixels.size == h * w * c) { "expected ${h * w * c} pixel values, not ${pixels.size}" }
        val out = FloatArray(c * h * w)
        for (y in 0 until h) {
            for (x in 0 until w) {
                for (ch in 0 until c) {
                    var v = (pixels[(y * w + x) * c + ch].toInt() and 0xff).toFloat() / 255f
                    if (shape.mean != null) v = (v - shape.mean[ch]) / shape.std!![ch]
                    out[(ch * h + y) * w + x] = v
                }
            }
        }
        return out
    }
}
