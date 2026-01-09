#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for data mapping utilities.
"""

import pytest

from openjudge.utils.grader_info import get_all_grader_info

# from rm_gallery.core.utils.grader_info import get_all_grader_info


@pytest.mark.unit
class TestGraderInfoUtil:
    """Test cases for grader_info util."""

    def test_get_graders_info(self):
        """Test get_all_graders_info unti function."""
        all_grader_info = get_all_grader_info()

        assert len(all_grader_info) > 0, all_grader_info
        for gi in all_grader_info:
            assert len(gi.get("class_name")) > 0, gi
            assert len(gi.get("parent_class_names")) > 0, gi
            assert isinstance(gi.get("parent_class_names"), list), gi
            assert len(gi.get("init_method")) > 0, gi
            assert len(gi.get("aevaluate_method")) > 0, gi


if __name__ == "__main__":
    pytest.main([__file__])
