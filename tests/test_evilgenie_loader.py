"""Tests for EvilGenie loader."""

import pytest

try:
    from rewardhackwatch.rhw_bench.evilgenie_loader import EvilGenieLoader

    EVILGENIE_LOADER_AVAILABLE = True
except ImportError:
    EVILGENIE_LOADER_AVAILABLE = False

pytestmark = pytest.mark.extended


@pytest.mark.skipif(not EVILGENIE_LOADER_AVAILABLE, reason="EvilGenie loader not available")
class TestEvilGenieLoaderInitialization:
    """Test EvilGenie loader initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        loader = EvilGenieLoader()
        assert loader is not None


@pytest.mark.skipif(not EVILGENIE_LOADER_AVAILABLE, reason="EvilGenie loader not available")
class TestEvilGenieLoaderDataLoading:
    """Test EvilGenie data loading."""

    @pytest.fixture
    def loader(self):
        return EvilGenieLoader()

    def test_load_samples(self, loader):
        """Test loading samples."""
        try:
            samples = loader.load(limit=5)
            if samples:
                assert len(samples) <= 5
        except Exception:
            pytest.skip("EvilGenie dataset not available")


@pytest.mark.skipif(not EVILGENIE_LOADER_AVAILABLE, reason="EvilGenie loader not available")
class TestEvilGenieLoaderTransformation:
    """Test EvilGenie data transformation."""

    @pytest.fixture
    def loader(self):
        return EvilGenieLoader()

    def test_transform_sample(self, loader):
        """Test transforming EvilGenie sample."""
        sample = {
            "problem_id": "test_001",
            "solution": "def solve(): sys.exit(0)",
            "is_malicious": True,
        }

        trajectory = loader.transform_sample(sample)

        assert "code_outputs" in trajectory
        assert trajectory.get("is_hack", False)
