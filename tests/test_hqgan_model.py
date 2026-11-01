import unittest
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from augmentation.hqgan.model import (haar, inverse_haar, HSC, AHD, Generator,
                                      Discriminator, discriminator_loss, generator_loss)


class HQGANModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_haar_roundtrip_and_gradients(self):
        x = torch.randn(2,3,16,16, requires_grad=True)
        restored = inverse_haar(haar(x))
        torch.testing.assert_close(restored, x)
        restored.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        _, *details = haar(torch.ones(1,1,8,8))
        self.assertTrue(all(torch.count_nonzero(v) == 0 for v in details))

    def test_hsc_has_gradient_to_both_paths(self):
        low = torch.randn(2,8,8,8, requires_grad=True)
        high = torch.randn(2,4,64,64, requires_grad=True)
        HSC(8,4)(low,high).square().mean().backward()
        self.assertGreater(float(low.grad.abs().sum()), 0)
        self.assertGreater(float(high.grad.abs().sum()), 0)

    def test_literal_signed_pooling_cancels_high_frequency_signal(self):
        x = torch.randn(1,2,16,16)
        ll, lh, hl, hh = haar(x)
        details = inverse_haar((torch.zeros_like(ll), lh, hl, hh))
        pooled = torch.nn.functional.adaptive_avg_pool2d(details,4)
        torch.testing.assert_close(pooled,torch.zeros_like(pooled),atol=1e-7,rtol=0)

    def test_ahd_controller_interval_bounds_and_checkpoint_state(self):
        a = AHD(initial_steps=10, gain=10)
        real, fake = torch.ones(2,64), -torch.ones(2,64)
        for _ in range(3): a.update(real,fake)
        self.assertEqual(float(a.horizon),10)
        a.update(real,fake)
        self.assertAlmostEqual(float(a.horizon),14)
        b = AHD(initial_steps=10, gain=10)
        b.load_state_dict(a.state_dict())
        self.assertEqual(float(b.horizon),14)
        a.horizon.fill_(500)
        for _ in range(4): a.update(real,fake)
        self.assertEqual(float(a.horizon),500)
        a.horizon.fill_(1)
        for _ in range(4): a.update(-real,-fake)
        self.assertEqual(float(a.horizon),1)
        x = torch.ones(2,4,8,8, requires_grad=True)
        a(x).mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_gan_update_changes_generator_and_discriminator(self):
        for resolution in (64,128):
            with self.subTest(resolution=resolution):
                g = Generator(resolution=resolution, width=1/32)
                d = Discriminator(width=1/32)
                fake = g(torch.randn(2,256))
                self.assertEqual(fake.shape, (2,3,resolution,resolution))
                self.assertTrue((fake.abs() <= 1).all())
                real_logits, fake_logits = d(torch.randn_like(fake)), d(fake.detach())
                self.assertEqual(real_logits.shape,(2,64))
                od = torch.optim.Adam(d.parameters(),lr=.0002)
                before = next(d.parameters()).detach().clone()
                discriminator_loss(real_logits,fake_logits).backward(); od.step()
                self.assertFalse(torch.equal(before,next(d.parameters())))
                d.requires_grad_(False)
                og = torch.optim.Adam(g.parameters(),lr=.0002)
                before = next(g.parameters()).detach().clone()
                generator_loss(d(fake)).backward(); og.step()
                self.assertFalse(torch.equal(before,next(g.parameters())))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in g.parameters()))

    def test_losses_are_stable_at_extreme_logits(self):
        x = torch.tensor([-1000.,1000.],requires_grad=True)
        loss = discriminator_loss(x,-x)+generator_loss(x)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__': unittest.main()
