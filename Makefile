.PHONY: install test run audit clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest tests

run:
	python -m src.pipeline

audit:
	python -m src.audit

clean:
	python -c "import shutil, pathlib; data=pathlib.Path('data'); [shutil.rmtree(p, ignore_errors=True) if p.is_dir() else (p.unlink() if p.name != '.gitkeep' else None) for p in data.glob('*')]; shutil.rmtree('.pytest_cache', ignore_errors=True); [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
