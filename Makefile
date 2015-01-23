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
	
update_server:
	@echo "Updating server"
	@echo "==============="
	git archive master --format tar | tar -t | rsync -avz --no-dirs --rsh='ssh ssh.cs.kuleuven.be ssh -p 2222' --chmod=u=rwx,g=rx,o=rx --files-from=- ./ problog@adams.cs.kuleuven.be:/home/problog/problog2.1
	rsync -avzr --exclude '*.DS_Store' --exclude '*.swp' --exclude '*~' --rsh='ssh ssh.cs.kuleuven.be ssh -p 2222' --chmod=u=rwx,g=rx,o=rx --exclude '*~' ./web/js/ problog@adams.cs.kuleuven.be:/home/problog/public_html/js
	rsync -avzr --chmod=u=rwx,g=rx,o=rx ./web/editor.html ssh.cs.kuleuven.be:/cw/vwww/dtai/public_html/problog/
	ssh ssh.cs.kuleuven.be ssh -p 2222 problog@adams.cs.kuleuven.be python /home/problog/problog2.1/setup.py
	ssh ssh.cs.kuleuven.be ssh -p 2222 problog@adams.cs.kuleuven.be python3 /home/problog/problog2.1/setup.py
	@echo "======================================================================"

