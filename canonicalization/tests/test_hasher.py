"""Tests for content hasher."""

import pytest
from discovery.types import VenueType
from canonicalization.hasher import ContentHasher


class TestTextNormalization:
    """Tests for text normalization."""
    
    def test_normalize_line_endings(self):
        """Test line ending normalization."""
        text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        normalized = ContentHasher.normalize_text(text)
        
        # Should only have \n
        assert '\r' not in normalized
        assert '\r\n' not in normalized
    
    def test_normalize_trailing_whitespace(self):
        """Test trailing whitespace removal."""
        text = "Line 1   \nLine 2\t\nLine 3"
        normalized = ContentHasher.normalize_text(text)
        
        # Trailing whitespace should be removed
        lines = normalized.split('\n')
        assert all(not line.endswith(' ') and not line.endswith('\t') for line in lines)
    
    def test_normalize_multiple_blank_lines(self):
        """Test collapsing multiple blank lines."""
        text = "Line 1\n\n\nLine 2\n\nLine 3"
        normalized = ContentHasher.normalize_text(text)
        
        # Should not have more than one consecutive blank line
        assert '\n\n\n' not in normalized
    
    def test_normalize_trailing_blank_lines(self):
        """Test removal of trailing blank lines."""
        text = "Line 1\nLine 2\n\n\n"
        normalized = ContentHasher.normalize_text(text)
        
        # Should not end with blank lines
        assert not normalized.endswith('\n') or normalized.endswith('\n\n')
        # Actually, it should end with content
        assert normalized.strip() == normalized.rstrip()
    
    def test_normalize_empty_text(self):
        """Test normalization of empty text."""
        normalized = ContentHasher.normalize_text("")
        assert normalized == ""
    
    def test_normalize_whitespace_only(self):
        """Test normalization of whitespace-only text."""
        normalized = ContentHasher.normalize_text("   \n\n\t\t  \n")
        assert normalized == ""
    
    def test_normalize_unicode(self):
        """Test normalization preserves unicode characters."""
        text = "Will Bitcoin reach $100,000? 🚀\nÉlection 2024"
        normalized = ContentHasher.normalize_text(text)
        
        assert "🚀" in normalized
        assert "Élection" in normalized


class TestContentHash:
    """Tests for content hash generation."""
    
    def test_hash_stability(self):
        """Test that same text produces same hash."""
        text = "Market Statement:\nWill Bitcoin reach $100,000?"
        
        hash1 = ContentHasher.hash_content(text)
        hash2 = ContentHasher.hash_content(text)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length
    
    def test_hash_different_texts(self):
        """Test that different texts produce different hashes."""
        text1 = "Market Statement:\nWill Bitcoin reach $100,000?"
        text2 = "Market Statement:\nWill Bitcoin reach $200,000?"
        
        hash1 = ContentHasher.hash_content(text1)
        hash2 = ContentHasher.hash_content(text2)
        
        assert hash1 != hash2
    
    def test_hash_normalization_consistency(self):
        """Test that normalized versions produce same hash."""
        text1 = "Line 1\nLine 2"
        text2 = "Line 1\r\nLine 2"  # Different line endings
        text3 = "Line 1   \nLine 2"  # Trailing whitespace
        
        hash1 = ContentHasher.hash_content(text1)
        hash2 = ContentHasher.hash_content(text2)
        hash3 = ContentHasher.hash_content(text3)
        
        # After normalization, should be same
        assert hash1 == hash2
        assert hash1 == hash3
    
    def test_hash_empty_text(self):
        """Test hashing empty text."""
        hash_val = ContentHasher.hash_content("")
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
    
    @pytest.mark.asyncio
    async def test_hash_async(self):
        """Test async hash method."""
        text = "Test market statement"
        hash_val = await ContentHasher.hash_content_async(text)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
    
    @pytest.mark.asyncio
    async def test_hash_batch(self):
        """Test batch hashing (non-blocking)."""
        texts = [
            "Market 1 statement",
            "Market 2 statement",
            "Market 3 statement",
        ]
        
        hashes = await ContentHasher.hash_batch(texts)
        
        assert len(hashes) == 3
        assert all(isinstance(h, str) for h in hashes)
        assert all(len(h) == 64 for h in hashes)
        # All should be different
        assert len(set(hashes)) == 3


class TestIdentityHash:
    """Tests for identity hash generation."""
    
    def test_identity_hash_format(self):
        """Test identity hash format."""
        hash_val = ContentHasher.identity_hash(VenueType.KALSHI, "MARKET-123")
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA-256 hex digest
    
    def test_identity_hash_consistency(self):
        """Test identity hash is consistent."""
        hash1 = ContentHasher.identity_hash(VenueType.KALSHI, "MARKET-123")
        hash2 = ContentHasher.identity_hash(VenueType.KALSHI, "MARKET-123")
        
        assert hash1 == hash2
    
    def test_identity_hash_different_venues(self):
        """Test different venues produce different hashes."""
        hash1 = ContentHasher.identity_hash(VenueType.KALSHI, "MARKET-123")
        hash2 = ContentHasher.identity_hash(VenueType.POLYMARKET, "MARKET-123")
        
        assert hash1 != hash2
    
    def test_identity_hash_different_markets(self):
        """Test different markets produce different hashes."""
        hash1 = ContentHasher.identity_hash(VenueType.KALSHI, "MARKET-123")
        hash2 = ContentHasher.identity_hash(VenueType.KALSHI, "MARKET-456")
        
        assert hash1 != hash2


class TestHashChangeDetection:
    """Tests for change detection using content hash."""
    
    def test_detect_content_change(self):
        """Test that content changes produce different hashes."""
        original = "Market Statement:\nWill Bitcoin reach $100,000?"
        updated = "Market Statement:\nWill Bitcoin reach $100,000 by Dec 31, 2024?"
        
        hash_original = ContentHasher.hash_content(original)
        hash_updated = ContentHasher.hash_content(updated)
        
        assert hash_original != hash_updated
    
    def test_detect_no_change(self):
        """Test that unchanged content produces same hash."""
        text = "Market Statement:\nWill Bitcoin reach $100,000?"
        
        hash1 = ContentHasher.hash_content(text)
        # Simulate re-processing same text
        hash2 = ContentHasher.hash_content(text)
        
        assert hash1 == hash2

