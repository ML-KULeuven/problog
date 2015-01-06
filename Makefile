.PHONY: default test test2 test3 docs deploy_docs

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

show_package:
	@echo "Listing package content"
	@echo "======================="
	git archive master --format tar | tar -t
	@echo "======================================================================"
	
package: docs
	@echo "Creating package"
	@echo "================"
	rm -f problog.zip
	git archive master --format zip --prefix problog2.1/ > problog.zip
	mkdir -p problog2.1
	ln -s ../docs problog2.1/docs
	zip -r problog.zip problog2.1/docs/build/
	rm problog2.1/docs
	rmdir problog2.1	
	@echo "======================================================================"
	
