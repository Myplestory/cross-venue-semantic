"""Edge case tests for content hasher."""

import pytest
from canonicalization.hasher import ContentHasher


class TestNormalizationEdgeCases:
    """Tests for edge cases in text normalization."""
    
    def test_normalize_only_whitespace(self):
        """Test normalization of text with only whitespace."""
        text = "   \n\n\t\t  \n"
        normalized = ContentHasher.normalize_text(text)
        
        assert normalized == ""
    
    def test_normalize_only_newlines(self):
        """Test normalization of text with only newlines."""
        text = "\n\n\n"
        normalized = ContentHasher.normalize_text(text)
        
        assert normalized == ""
    
    def test_normalize_mixed_line_endings(self):
        """Test normalization with mixed line endings."""
        text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        normalized = ContentHasher.normalize_text(text)
        
        # Should only have \n
        assert '\r' not in normalized
        assert '\r\n' not in normalized
        assert normalized.count('\n') == 3  # 3 line breaks
    
    def test_normalize_trailing_spaces_per_line(self):
        """Test normalization removes trailing spaces per line."""
        text = "Line 1   \nLine 2\t\t\nLine 3"
        normalized = ContentHasher.normalize_text(text)
        
        lines = normalized.split('\n')
        assert all(not line.endswith(' ') for line in lines)
        assert all(not line.endswith('\t') for line in lines)
    
    def test_normalize_preserves_leading_spaces(self):
        """Test normalization preserves leading spaces (indentation)."""
        text = "  Indented line\n    More indented"
        normalized = ContentHasher.normalize_text(text)
        
        # Leading spaces should be preserved
        assert normalized.startswith("  Indented")
        assert "    More indented" in normalized
    
    def test_normalize_collapses_multiple_blanks(self):
        """Test normalization collapses multiple blank lines."""
        text = "Line 1\n\n\n\nLine 2"
        normalized = ContentHasher.normalize_text(text)
        
        # Should not have more than one consecutive blank line
        assert '\n\n\n' not in normalized
        # But can have one blank line
        assert '\n\n' in normalized or '\n' in normalized
    
    def test_normalize_unicode_whitespace(self):
        """Test normalization handles unicode whitespace."""
        text = "Line 1\u2003\u2004Line 2"  # Unicode spaces
        normalized = ContentHasher.normalize_text(text)
        
        # Should preserve unicode whitespace (not standard spaces)
        assert "Line 1" in normalized
        assert "Line 2" in normalized
    
    def test_normalize_very_long_text(self):
        """Test normalization of very long text."""
        long_text = "Line " + "x" * 10000 + "\n" + "Line " + "y" * 10000
        normalized = ContentHasher.normalize_text(long_text)
        
        assert len(normalized) > 0
        assert "Line" in normalized
    
    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        normalized = ContentHasher.normalize_text("")
        assert normalized == ""
    
    def test_normalize_single_line(self):
        """Test normalization of single line."""
        text = "Single line with no newlines"
        normalized = ContentHasher.normalize_text(text)
        
        assert normalized == text
    
    def test_normalize_only_trailing_newlines(self):
        """Test normalization removes trailing newlines."""
        text = "Content\n\n\n"
        normalized = ContentHasher.normalize_text(text)
        
        assert not normalized.endswith('\n\n')
        assert normalized.endswith("Content") or normalized == "Content"


class TestHashEdgeCases:
    """Tests for edge cases in hashing."""
    
    def test_hash_empty_string(self):
        """Test hashing empty string."""
        hash_val = ContentHasher.hash_content("")
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA-256 hex digest
        # Empty string should always produce same hash
        assert hash_val == ContentHasher.hash_content("")
    
    def test_hash_very_long_text(self):
        """Test hashing very long text."""
        long_text = "A" * 100000  # 100KB text
        hash_val = ContentHasher.hash_content(long_text)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
    
    def test_hash_unicode_text(self):
        """Test hashing unicode text."""
        unicode_text = "Will Bitcoin reach $100,000? 🚀 Élection 2024 中文 🎉"
        hash_val = ContentHasher.hash_content(unicode_text)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
        # Same text should produce same hash
        assert hash_val == ContentHasher.hash_content(unicode_text)
    
    def test_hash_special_characters(self):
        """Test hashing text with special characters."""
        special_text = "Test: $100,000 & €90,000 (Yes/No) [2024] {BTC}"
        hash_val = ContentHasher.hash_content(special_text)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
    
    def test_hash_binary_like_text(self):
        """Test hashing text that looks like binary."""
        binary_like = "\x00\x01\x02\x03\x04\x05"
        hash_val = ContentHasher.hash_content(binary_like)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
    
    def test_hash_case_sensitivity(self):
        """Test that hash is case-sensitive."""
        text1 = "Will Bitcoin reach $100,000?"
        text2 = "WILL BITCOIN REACH $100,000?"
        
        hash1 = ContentHasher.hash_content(text1)
        hash2 = ContentHasher.hash_content(text2)
        
        # Different case should produce different hash
        assert hash1 != hash2
    
    def test_hash_whitespace_sensitivity(self):
        """Test that hash is sensitive to whitespace differences."""
        text1 = "Line 1\nLine 2"
        text2 = "Line 1\n\nLine 2"  # Extra newline
        
        hash1 = ContentHasher.hash_content(text1)
        hash2 = ContentHasher.hash_content(text2)
        
        # Different whitespace should produce different hash
        assert hash1 != hash2
    
    def test_hash_normalization_consistency(self):
        """Test that normalized versions produce same hash."""
        text1 = "Line 1   \nLine 2"  # Trailing spaces
        text2 = "Line 1\nLine 2"     # No trailing spaces
        
        # After normalization, should be same
        norm1 = ContentHasher.normalize_text(text1)
        norm2 = ContentHasher.normalize_text(text2)
        
        hash1 = ContentHasher.hash_content(norm1)
        hash2 = ContentHasher.hash_content(norm2)
        
        # Normalized versions should produce same hash
        assert hash1 == hash2


class TestIdentityHashEdgeCases:
    """Tests for edge cases in identity hashing."""
    
    def test_identity_hash_empty_market_id(self):
        """Test identity hash with empty market ID."""
        from discovery.types import VenueType
        
        hash_val = ContentHasher.identity_hash(VenueType.KALSHI, "")
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
    
    def test_identity_hash_very_long_market_id(self):
        """Test identity hash with very long market ID."""
        from discovery.types import VenueType
        
        long_id = "A" * 1000
        hash_val = ContentHasher.identity_hash(VenueType.KALSHI, long_id)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
    
    def test_identity_hash_special_characters(self):
        """Test identity hash with special characters in market ID."""
        from discovery.types import VenueType
        
        special_id = "MARKET-123:test@example.com"
        hash_val = ContentHasher.identity_hash(VenueType.KALSHI, special_id)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
    
    def test_identity_hash_unicode_market_id(self):
        """Test identity hash with unicode market ID."""
        from discovery.types import VenueType
        
        unicode_id = "MARKET-中文-🚀"
        hash_val = ContentHasher.identity_hash(VenueType.KALSHI, unicode_id)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64


class TestAsyncHashEdgeCases:
    """Tests for async hashing edge cases."""
    
    @pytest.mark.asyncio
    async def test_hash_async_empty_string(self):
        """Test async hash with empty string."""
        hash_val = await ContentHasher.hash_content_async("")
        
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
    
    @pytest.mark.asyncio
    async def test_hash_batch_empty_list(self):
        """Test batch hashing with empty list."""
        hashes = await ContentHasher.hash_batch([])
        
        assert hashes == []
    
    @pytest.mark.asyncio
    async def test_hash_batch_large_list(self):
        """Test batch hashing with large list (1000 texts)."""
        texts = [f"Market {i} statement" for i in range(1000)]
        
        hashes = await ContentHasher.hash_batch(texts)
        
        assert len(hashes) == 1000
        assert all(isinstance(h, str) for h in hashes)
        assert all(len(h) == 64 for h in hashes)
        # All should be unique
        assert len(set(hashes)) == 1000
    
    @pytest.mark.asyncio
    async def test_hash_batch_mixed_content(self):
        """Test batch hashing with mixed content types."""
        texts = [
            "Normal text",
            "",  # Empty
            "A" * 10000,  # Very long
            "Unicode: 🚀 中文",
            "Special: $100,000 & €90,000",
        ]
        
        hashes = await ContentHasher.hash_batch(texts)
        
        assert len(hashes) == 5
        assert all(isinstance(h, str) for h in hashes)
        assert all(len(h) == 64 for h in hashes)
        # All should be different
        assert len(set(hashes)) == 5
    
    @pytest.mark.asyncio
    async def test_hash_batch_duplicate_texts(self):
        """Test batch hashing with duplicate texts."""
        texts = ["Same text"] * 10
        
        hashes = await ContentHasher.hash_batch(texts)
        
        assert len(hashes) == 10
        # All should be same hash
        assert len(set(hashes)) == 1
        assert all(h == hashes[0] for h in hashes)


class TestHashCollisionResistance:
    """Tests for hash collision resistance."""
    
    def test_different_texts_different_hashes(self):
        """Test that different texts produce different hashes."""
        texts = [
            "Text 1",
            "Text 2",
            "Text 3",
            "Different text",
            "Another different text",
        ]
        
        hashes = [ContentHasher.hash_content(text) for text in texts]
        
        # All should be different
        assert len(set(hashes)) == len(texts)
    
    def test_similar_texts_different_hashes(self):
        """Test that similar texts produce different hashes."""
        texts = [
            "Will Bitcoin reach $100,000?",
            "Will Bitcoin reach $100,000",
            "Will Bitcoin reach $100,000.",
            "Will Bitcoin reach $100,000!",
        ]
        
        hashes = [ContentHasher.hash_content(text) for text in texts]
        
        # All should be different (even small differences)
        assert len(set(hashes)) == len(texts)


