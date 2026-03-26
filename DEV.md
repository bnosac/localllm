### Contribute

Run pytest locally

```
pip install uv
uv run pytest -ra -q --doctest-modules
```

Run pytest on small set of changes

```
pip uninstall -y localllm
pip install localllm
pytest -ra -q --doctest-modules --ignore=dist --ignore=build
pytest src/localllm/dspy.py
pytest src/localllm/utils.py
pytest src/localllm/config.py
```

Make sure the source code is formatted

```
pip install pre-commit
pre-commit run --all-files
```


