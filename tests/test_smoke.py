import unittest

import numpy as np

from nultra_multiband_ecosystem import processor as core


class ProcessorSmokeTests(unittest.TestCase):
    def test_render_multiband_preserves_stereo_shape(self) -> None:
        old_buffer = core.CHAOS_BUFFER_LENGTH
        old_burn_in = core.CHAOS_BURN_IN
        try:
            core.CHAOS_BUFFER_LENGTH = 16384
            core.CHAOS_BURN_IN = 512
            sample_rate, audio = core.generate_sine_wave(
                frequency_hz=440.0,
                duration_seconds=0.5,
                sample_rate=12000,
                amplitude=0.4,
                channels=2,
            )
            processed, stats = core.render_multiband(
                audio=audio,
                sample_rate=sample_rate,
                low_xover_hz=core.LOW_CROSSOVER_HZ,
                high_xover_hz=core.HIGH_CROSSOVER_HZ,
                wet_dry=1.0,
            )
        finally:
            core.CHAOS_BUFFER_LENGTH = old_buffer
            core.CHAOS_BURN_IN = old_burn_in

        self.assertEqual(processed.shape, audio.shape)
        self.assertEqual(processed.ndim, 2)
        self.assertTrue(np.isfinite(processed).all())
        self.assertIn("low", stats)
        self.assertIn("mid", stats)
        self.assertIn("high", stats)


if __name__ == "__main__":
    unittest.main()
