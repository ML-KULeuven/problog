.PHONY: default test test3 docs deploy_docs binaries clean-binaries

default: test docs
	
test: test3

test3:
	@echo "Running unit tests in python3"
	@echo "============================="
	python3 -m unittest discover -v
	@echo "======================================================================"
	@echo ""

test-lficont:
	@echo "Running unit tests in python3"
	@echo "============================="
	python3 -m unittest discover -v -p test_lficont*.py
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

# NOTE: `deploy` and `deploy_dev` publish an sdist only, which since the move to
# per-platform wheels is a PURE package with no solver binaries.  Publishing a
# usable release means the release workflow, which builds a wheel per platform.
# These targets are kept for the development-release flow only.

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
	python3 -m build --sdist
	twine upload dist/*
	# git checkout develop
	# git merge master

deploy_dev: test3 incr_version_dev
	@read -r -p "WARNING: This will upload a new development release! Press ENTER to proceed, CTRL-C to cancel."
	git push
	rm -f dist/*
	python3 -m build --sdist
	twine upload dist/*

incr_version_dev:
	python3 tools/version_bump.py dev
	git commit -m "Deploy new development version." problog/version.py

incr_version_release:
	python3 tools/version_bump.py release
	git commit problog/version.py -m "Deploy new release version."


# ---------------------------------------------------------------------------
# Solver binaries
#
# The Python build is pure: it packages whatever is in problog/bin/<platform>/.
# `make binaries` is what puts it there.  Run it once in a checkout; the release
# workflow runs it in each platform job before building the wheel.
#
# Build the wheel and sdist as SEPARATE invocations:
#     python -m build --wheel     # picks up problog/bin/<platform>/
#     python -m build --sdist     # pure; MANIFEST.in excludes the binaries
# Plain `python -m build` builds the sdist first and then the wheel from that
# sdist, so the wheel would inherit MANIFEST.in's exclusions and ship empty.
# ---------------------------------------------------------------------------

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
  PLAT := darwin
  EXE  :=
  # One clang invocation with both -arch flags gives a universal2 binary; no
  # lipo step, and no architecture-matched Homebrew (which only gmp needs).
  ARCHES := -arch x86_64 -arch arm64
  CC := clang
  DSHARP_FLAGS := CXX=clang++ LINK=clang++ \
                  CXXFLAGS="-pipe -O3 -w $(ARCHES)" LFLAGS="$(ARCHES)"
else
  ifneq (,$(filter MINGW% MSYS%,$(UNAME_S)))
    PLAT := windows
    EXE  := .exe
  else
    PLAT := linux
    EXE  :=
  endif
  ARCHES :=
  CC := gcc
  DSHARP_FLAGS :=
endif

BINDIR      := problog/bin/$(PLAT)
DSHARP      := $(BINDIR)/dsharp$(EXE)
MAXSATZ     := $(BINDIR)/maxsatz$(EXE)
DSHARP_SRC  := extern/dsharp
MAXSATZ_SRC := problog/bin/source/maxsatz/maxsatz2009.c
WORK        := build/dsharp

DSHARP_SOURCES := $(shell find $(DSHARP_SRC)/src -type f 2>/dev/null)

binaries: $(DSHARP) $(MAXSATZ)

$(BINDIR):
	mkdir -p $@

# dsharp's Makefile writes object files next to its sources (OBJECTS_DIR is
# decorative -- the rules hardcode `-o Basics.o`), so build in a copy and leave
# the submodule clean.
$(DSHARP): $(DSHARP_SOURCES) | $(BINDIR)
	@test -f $(DSHARP_SRC)/Makefile_nogmp || { \
	  echo "extern/dsharp is empty -- run: git submodule update --init --recursive"; \
	  exit 1; }
	rm -rf $(WORK)
	mkdir -p $(dir $(WORK))
	cp -R $(DSHARP_SRC) $(WORK)
	cp $(WORK)/Makefile_nogmp $(WORK)/Makefile
	$(MAKE) -C $(WORK) all $(DSHARP_FLAGS)
	cp $(WORK)/dsharp$(EXE) $@ 2>/dev/null || cp $(WORK)/dsharp $@
	chmod 755 $@

$(MAXSATZ): $(MAXSATZ_SRC) | $(BINDIR)
	$(CC) $(ARCHES) -O2 -o $@ $<
	chmod 755 $@

clean-binaries:
	rm -rf $(WORK) $(DSHARP) $(MAXSATZ)
