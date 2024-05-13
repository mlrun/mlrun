PACKAGES := mlrun-pipelines-kfp-common mlrun-pipelines-kfp-v1-8 mlrun-pipelines-kfp-v2

TWINE_USERNAME ?= __token__
TWINE_PASSWORD ?= "replace with your token"

.PHONY: build release clean

build:
	@for pkg in $(PACKAGES); do \
        echo "Building $$pkg"; \
        cd $$pkg && python -m build; \
        cd ..; \
    done

release:
	@for pkg in $(PACKAGES); do \
        echo "Releasing $$pkg to PyPI"; \
        cd $$pkg && TWINE_USERNAME=$(TWINE_USERNAME) TWINE_PASSWORD=$(TWINE_PASSWORD) twine upload dist/*; \
        cd ..; \
    done

clean:
	@for pkg in $(PACKAGES); do \
        echo "Cleaning $$pkg" \
        cd $$pkg && rm -rf dist build *.egg-info; \
        cd ..; \
	done
