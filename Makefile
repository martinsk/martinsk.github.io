.PHONY: build drafts serve clean

build:
	emacs -Q --batch -l build.el -f site-build

# Like build, but also includes posts not yet tagged :published:
drafts:
	DRAFTS=1 emacs -Q --batch -l build.el -f site-build

serve: build
	python3 -m http.server --directory _site 8000

clean:
	rm -rf _site .build
