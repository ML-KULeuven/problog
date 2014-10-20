.PHONY: default test test2 test3 docs deploy_docs

DOCS_LOCATION=ssh.cs.kuleuven.be:/cw/wwwserver/extern2/people/anton.dries/public_html/problog/

default: test docs
	
test: test2 test3

test2:
	@echo "Running unit tests in python2"
	@echo "============================="
	python -m unittest discover -v
	@echo "======================================================================"
	@echo ""
	
test3:
	@echo "Running unit tests in python3"
	@echo "============================="
	python3 -m unittest discover -v
	@echo "======================================================================"
	@echo ""
	
docs:
	@echo "Creating HTML documentation"
	@echo "==========================="
	make -C docs/ html
	@echo "======================================================================"
	
deploy_docs:
	@echo "Uploading docs"
	@echo "=============="
	rsync --archive docs/build/html/* ${DOCS_LOCATION}
	@echo "======================================================================"