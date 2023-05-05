# Copyright 2021-2022 The DADApy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test the README content."""

import inspect
import json
import os
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import mistletoe
from mistletoe.ast_renderer import ASTRenderer

MISTLETOE_CODE_BLOCK_ID = "CodeFence"
CUR_PATH = Path(inspect.getfile(inspect.currentframe())).parent
ROOT_DIR = Path(CUR_PATH, "../..").resolve().absolute()


def code_block_filter(block_dict: Dict, language: Optional[str] = None) -> bool:
    """
    Check Mistletoe block is a code block.

    Args:
        block_dict: the block dictionary describing a Mistletoe document node
        language: optionally check also the language field

    Returns:
        True if the block satistifes the conditions, False otherwise
    """
    return block_dict["type"] == MISTLETOE_CODE_BLOCK_ID and (
        language is None or block_dict["language"] == language
    )


def python_code_block_filter(block_dict: Dict) -> bool:
    """Filter Python code blocks."""
    return code_block_filter(block_dict, language="python")


def code_block_extractor(child_dict: Dict) -> str:
    """Extract Mistletoe code block from Mistletoe CodeFence child."""
    # we assume that 'children' of CodeFence child has only one child (may be wrong)
    assert len(child_dict["children"]) == 1
    return child_dict["children"][0]["content"]


class BaseTestMarkdownDocs:
    """Base test class for testing Markdown documents."""

    DOC_PATH: Path
    blocks: List[Dict]
    code_blocks: List[str]
    python_blocks: List[str]

    @classmethod
    def setup_class(cls) -> None:
        """Set up the test."""
        doc_content = cls.DOC_PATH.read_text()
        doc_file_descriptor = StringIO(doc_content)
        markdown_parsed = mistletoe.markdown(doc_file_descriptor, renderer=ASTRenderer)
        markdown_json = json.loads(markdown_parsed)
        cls.blocks = markdown_json["children"]
        cls.code_blocks = list(
            map(code_block_extractor, filter(code_block_filter, cls.blocks))
        )
        cls.python_blocks = list(
            map(code_block_extractor, filter(python_code_block_filter, cls.blocks))
        )


class TestReadme(BaseTestMarkdownDocs):
    """Test the README Python code snippets."""

    DOC_PATH = Path(ROOT_DIR, "README.md")

    def test_quickstart_code(self) -> None:
        """Test the Python code in the readme file."""
        quickstart_code_snippet = self.python_blocks[0]

        with open("_test_readme_temp.py", "w") as f:
            for line in quickstart_code_snippet.splitlines():
                f.write(line)
                f.write("\n")

        termination_code = os.system("python _test_readme_temp.py")
        os.remove("_test_readme_temp.py")

        assert termination_code == 0
