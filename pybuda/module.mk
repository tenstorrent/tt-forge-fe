include pybuda/csrc/module.mk

$(PYTHON_ENV_ROOT)/lib/$(PYTHON_VERSION)/site-packages/pybuda.egg-link: $(PYTHON_ENV) $(PYTHON_ENV_ROOT)/lib/$(PYTHON_VERSION)/site-packages/pybuda/_C.so
	bash -c "source $(PYTHON_ENV_ROOT)/bin/activate; cd pybuda; pip install -e ."
	touch -r $(PYTHON_ENV_ROOT)/lib/$(PYTHON_VERSION)/site-packages/pybuda/_C.so $@

pybuda: pybuda/csrc $(PYTHON_ENV_ROOT)/lib/$(PYTHON_VERSION)/site-packages/pybuda.egg-link ;

