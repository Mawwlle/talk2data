fmt:
	poetry run ruff check --fix .
	poetry run black .

lint:
	poetry run ruff check .
	poetry run mypy .

test:
	poetry run pytest -q
