### Contribute

Run pytest locally

```
pip install uv
uv run pytest -ra -q --doctest-modules
```

Run pytest on small set of changes

```
pip install localllm
pytest -ra -q --doctest-modules
pytest localllm/src/localllm/dspy.py
```

Make sure the source code is formatted

```
pip install pre-commit
pre-commit run --all-files
```


