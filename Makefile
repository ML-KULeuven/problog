.PHONY: default test test3 docs deploy_docs

default: test docs
	
test: test3

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

update_server_js:
	@echo "Updating javascript"
	@echo "==================="
	rsync -avzr --exclude '*.DS_Store' --exclude '*.swp' --exclude '*~' --rsh='ssh ssh.cs.kuleuven.be ssh -p 2222' --chmod=u=rwx,g=rx,o=rx --exclude '*~' ./problog/web/js/ problog@verne.cs.kuleuven.be:/home/problog/public_html/js
	rsync -avzr --chmod=u=rwx,g=rx,o=rx ./problog/web/editor.html ssh.cs.kuleuven.be:/cw/vwww1/dtai_static/public_html/problog/
	@echo "======================================================================"

update_server: 
	@echo "Updating server"
	@echo "==============="
	git archive master --format tar | tar -t | rsync -avz --no-dirs --rsh='ssh ssh.cs.kuleuven.be ssh -p 2222' --chmod=u=rwx,g=rx,o=rx --files-from=- ./ problog@verne.cs.kuleuven.be:/home/problog/problog2.1
	ssh ssh.cs.kuleuven.be ssh -p 2222 problog@verne.cs.kuleuven.be python3 /home/problog/problog2.1/problog-cli.py install
	ssh ssh.cs.kuleuven.be ssh -p 2222 problog@verne.cs.kuleuven.be python3 -m pip install /home/problog/problog2.1/.[sdd]
	@echo "======================================================================"

# prepare_deploy: test3
# 	git checkout master
# 	git merge develop --squash
# 	python -c 'import setup; setup.increment_version_release()'
# 	git add problog/version.py
# 	@echo "Next steps: git commit && make deploy"

deploy: test3 incr_version_release
	@read -r -p "WARNING: This will upload a new public release! Press ENTER to proceed, CTRL-C to cancel."
	# git checkout master
	git push
	# git push public master
	rm -f dist/*
	python setup.py sdist
	twine upload dist/*
	# git checkout develop
	# git merge master

deploy_dev: test3 incr_version_dev
	@read -r -p "WARNING: This will upload a new development release! Press ENTER to proceed, CTRL-C to cancel."
	git push
	rm -f dist/*
	python setup.py sdist
	twine upload dist/*

incr_version_dev:
	python -c 'import setup; setup.increment_version_dev()'
	git commit -m "Deploy new development version." problog/version.py

incr_version_release:
	python -c 'import setup; setup.increment_version_release()'
	git commit problog/version.py -m "Deploy new release version."
