"""Smoke tests to verify upgraded dependencies are compatible."""

import importlib

import pytest


class TestTorchCompat:
    """Verify torch 2.8.x works with our SB3 + gymnasium stack."""

    def test_torch_imports(self):
        import torch

        assert torch.__version__.startswith("2.8")

    def test_torch_tensor_ops(self):
        import torch

        t = torch.randn(4, 4)
        result = t @ t.T
        assert result.shape == (4, 4)

    def test_torch_nn_functional(self):
        """Ensure core nn.functional used by SB3 works."""
        import torch
        import torch.nn.functional as F

        logits = torch.randn(2, 3)
        probs = F.softmax(logits, dim=-1)
        assert probs.shape == (2, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2))

    def test_sb3_with_torch(self):
        """Verify stable-baselines3 can initialise a model on the new torch."""
        from stable_baselines3 import PPO

        model = PPO("MlpPolicy", "CartPole-v1", verbose=0, device="cpu")
        assert model.policy is not None

    def test_sb3_contrib_with_torch(self):
        """Verify sb3-contrib QRDQN works on the new torch."""
        from sb3_contrib import QRDQN

        model = QRDQN("MlpPolicy", "CartPole-v1", verbose=0, device="cpu")
        assert model.policy is not None


class TestSecurityPatchedPackages:
    """Verify patched package versions are installed."""

    @pytest.mark.parametrize(
        "package,min_version",
        [
            ("fonttools", "4.60.2"),
            ("filelock", "3.20.3"),
            ("jinja2", "3.1.6"),
            ("torch", "2.8.0"),
        ],
    )
    def test_minimum_patched_version(self, package, min_version):
        from packaging.version import Version

        mod = importlib.import_module(package if package != "fonttools" else "fontTools")
        installed = Version(mod.__version__)
        required = Version(min_version)
        assert installed >= required, (
            f"{package} {installed} < required minimum {required}"
        )

    def test_jinja2_sandbox_patch(self):
        """Confirm CVE-2025-27516 is mitigated: |attr('format') must not
        leak internal object state in a sandboxed environment."""
        from jinja2.sandbox import SandboxedEnvironment

        env = SandboxedEnvironment()
        # Pre-patch, this would leak the __globals__ dict of the lipsum
        # function via a format string obtained through |attr.
        template = env.from_string(
            "{{ (''|attr('format'))('{0.__class__.__init__.__globals__}', lipsum) }}"
        )
        result = template.render()
        # Patched version blocks the access (returns empty or raises)
        assert "__builtins__" not in result
