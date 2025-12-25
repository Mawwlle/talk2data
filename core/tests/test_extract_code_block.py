from hypothesis import given  # type: ignore[import-not-found]
from hypothesis import strategies as st  # type: ignore[import-not-found]

from core.workflow import extract_code_block


def test_extracts_python_code_block():
    text = """Here is code:
```python
print('hi')
```
"""
    assert extract_code_block(text) == "print('hi')"


def test_extracts_generic_code_block():
    text = """```\nSELECT * FROM table;\n```"""
    assert extract_code_block(text) == "SELECT * FROM table;"


def test_falls_back_to_plain_text():
    text = "just some text"
    assert extract_code_block(text) == "just some text"


@given(
    st.text(
        alphabet=st.characters(blacklist_characters="`"),
        min_size=0,
        max_size=200,
    )
)
def test_extracts_embedded_python_block(code):
    wrapped = f"```python\n{code}\n```"
    assert extract_code_block(wrapped) == code.strip()


@given(
    st.text(
        alphabet=st.characters(blacklist_characters="`"),
        min_size=0,
        max_size=200,
    )
)
def test_plain_text_passthrough(random_text):
    assert extract_code_block(random_text) == random_text.strip()
