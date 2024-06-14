# Every variable in subdir must be prefixed with subdir (emulating a namespace)
PYTHON_ENV_ROOT = $(OUT)/python_env
PYTHON_ENV = $(PYTHON_ENV_ROOT)/.installed

# Each module has a top level target as the entrypoint which must match the subdir name
python_env: $(PYTHON_ENV)/.installed

.PRECIOUS: $(PYTHON_ENV) $(PYTHON_ENV_ROOT)/%
$(PYTHON_ENV_ROOT)/.installed: python_env/requirements.txt
	$(PYTHON_VERSION) -m venv $(PYTHON_ENV_ROOT)
	bash -c "unset LD_PRELOAD; source $(PYTHON_ENV_ROOT)/bin/activate && python3 -m pip install --upgrade pip"
	bash -c "unset LD_PRELOAD; source $(PYTHON_ENV_ROOT)/bin/activate && pip3 install wheel==0.37.1"
	bash -c "unset LD_PRELOAD; source $(PYTHON_ENV_ROOT)/bin/activate && pip3 install -r python_env/requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html"
	touch $@

# If you depend on anything (headers, libs, etc) in the python env, build env first
$(PYTHON_ENV)/%: $(PYTHON_ENV) ;
