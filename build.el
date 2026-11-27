;;; build.el --- Build the site into _site/  -*- lexical-binding: t; -*-

;; Usage:  emacs -Q --batch -l build.el -f site-build     (or just `make')
;;
;; Layout:
;;   posts/SLUG.org   a post
;;   posts/SLUG/      its assets (images, source files, ...), optional
;;   index.html       hand-written front page; the Writing cards between the
;;                    "posts:begin" / "posts:end" comments are generated
;;
;; A post is built only if it is tagged as published:
;;
;;   #+FILETAGS: :published:
;;
;; It is then exported to _site/SLUG/index.html, posts/SLUG/ is copied next to
;; it, and it gets a card on the front page.  Anything else in posts/ is a
;; draft and is left out entirely; `make drafts' builds those too, for preview.
;;
;; Per-post metadata:
;;   #+TITLE:        page <h1>, <title>, card title
;;   #+SUBTITLE:     italic line under the title (org markup allowed)
;;   #+DESCRIPTION:  <meta name="description">
;;   #+DATE:         YYYY-MM-DD; front-page cards are sorted newest first
;;   #+KIND:         small label on the card (default "Notes")
;;   #+SUMMARY:      card text (org markup allowed; defaults to DESCRIPTION)
;; A heading with both :CUSTOM_ID: and :NAV: properties is listed in the nav
;; strip under the title, labelled with the :NAV: value.
;;
;; Inside a post, link to assets and other posts as they sit on disk, so the
;; links also work in Emacs:
;;   [[file:SLUG/figure.png]]        ->  figure.png  (same directory once built)
;;   [[file:other-post.org][text]]   ->  ../other-post/index.html

(require 'package)
(require 'ox)
(require 'ox-html)

(defconst site-root
  (file-name-directory (or load-file-name buffer-file-name)))
(defconst site-posts (expand-file-name "posts/" site-root))
(defconst site-out (expand-file-name "_site/" site-root))
(defconst site-static '("css" "assets" "CNAME")
  "Files and directories copied to the site unchanged.")
(defvar site-include-drafts (getenv "DRAFTS")
  "When non-nil, build unpublished posts as well.")
(defconst site-author "Martin S. Kristiansen")

;;; Dependencies -- htmlize gives src blocks their syntax-highlighting classes.

(setq package-user-dir (expand-file-name ".build/elpa" site-root))
(package-initialize)
(unless (package-installed-p 'htmlize)
  (package-refresh-contents)
  (package-install 'htmlize))

;;; Export settings

(setq make-backup-files nil
      org-export-use-babel nil              ; never execute code while building
      org-html-htmlize-output-type 'css     ; classes, coloured by css/main.css
      org-html-table-default-attributes nil
      org-html-text-markup-alist '((bold . "<strong>%s</strong>")
                                   (italic . "<em>%s</em>")
                                   (code . "<code>%s</code>")
                                   (verbatim . "<code>%s</code>")
                                   (strike-through . "<del>%s</del>")
                                   (underline . "<span class=\"underline\">%s</span>"))
      org-export-with-section-numbers nil
      org-export-with-toc nil
      org-export-with-sub-superscripts '{}  ; x_{i} and x^{2} only, never bare a_b
      org-html-doctype "html5"
      org-html-html5-fancy t)

;;; Page template

(defun site--head (info root)
  (let ((title (org-html-encode-plain-text
                (org-no-properties
                 (org-element-interpret-data (plist-get info :title)))))
        (description (plist-get info :description)))
    (concat
     "  <meta charset=\"utf-8\" />\n"
     "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
     (format "  <title>%s — %s</title>\n" title site-author)
     (when (org-string-nw-p description)
       (format "  <meta name=\"description\" content=\"%s\" />\n"
               (replace-regexp-in-string
                "\"" "&quot;" (org-html-encode-plain-text description))))
     "  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />\n"
     "  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />\n"
     "  <link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&amp;family=JetBrains+Mono:wght@300;400;500&amp;display=swap\" />\n"
     (format "  <link rel=\"stylesheet\" type=\"text/css\" href=\"%scss/main.css\" />\n" root)
     "  <script data-goatcounter=\"https://martinsk.goatcounter.com/count\"\n"
     "          async src=\"//gc.zgo.at/count.js\"></script>\n"
     ;; Only emitted when the post actually contains LaTeX.
     (org-html--build-mathjax-config info))))

(defun site--header (root)
  (format "  <header class=\"site-header\">
    <div class=\"container site-header__inner\">
      <a class=\"site-header__name\" href=\"%1$sindex.html\">%2$s</a>
      <nav class=\"site-nav\">
        <a href=\"%1$sindex.html#about\">About</a>
        <a href=\"%1$sindex.html#writing\">Writing</a>
        <a href=\"%1$sindex.html#projects\">Projects</a>
        <a href=\"https://github.com/martinsk\">GitHub</a>
      </nav>
    </div>
  </header>\n"
          root site-author))

(defun site--nav-strip (info root)
  "Links to the headings that carry a :NAV: label, or nil if there are none."
  (let ((links (org-element-map (plist-get info :parse-tree) 'headline
                 (lambda (h)
                   (let ((id (org-element-property :CUSTOM_ID h))
                         (label (org-element-property :NAV h)))
                     (when (and id label)
                       (format "        <a href=\"#%s\">%s</a>\n"
                               id (org-html-encode-plain-text label)))))
                 info)))
    (when links
      (concat "      <nav class=\"toc-strip\">\n"
              (format "        <a href=\"%sindex.html\">Home</a>\n" root)
              (apply #'concat links)
              "      </nav>\n"))))

(defun site-template (contents info)
  "Wrap the exported CONTENTS of a post in the site's page layout."
  (let ((root "../")                     ; every post lives at /SLUG/index.html
        (subtitle (plist-get info :subtitle)))
    (concat
     "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
     (site--head info root)
     "</head>\n<body>\n\n"
     (site--header root)
     "\n  <article>\n    <div class=\"container\">\n"
     "      <header>\n"
     (format "        <h1>%s</h1>\n" (org-export-data (plist-get info :title) info))
     "        <hr class=\"title-rule\" />\n"
     (when subtitle
       (format "        <p class=\"subtitle\">%s</p>\n" (org-export-data subtitle info)))
     "      </header>\n"
     (site--nav-strip info root)
     "      <main>\n" contents "      </main>\n"
     "    </div>\n  </article>\n\n"
     "  <footer>\n    <div class=\"container\">\n"
     (format "      <p>&copy; %s %s</p>\n" (format-time-string "%Y") site-author)
     "    </div>\n  </footer>\n\n</body>\n</html>\n")))

(defun site--fix-links (tree _backend info)
  "Rewrite file links in TREE from the source layout to the built layout."
  (let ((prefix (concat (file-name-base (plist-get info :input-file)) "/")))
    ;; The subtitle is parsed separately from the body, so cover it as well.
    (org-element-map (list tree (plist-get info :subtitle)) 'link
      (lambda (link)
        (when (equal (org-element-property :type link) "file")
          (let ((path (org-element-property :path link)))
            (cond
             ((string-prefix-p prefix path)                  ; own asset
              (org-element-put-property
               link :path (substring path (length prefix))))
             ((string-match "\\`\\([^/]+\\)\\.org\\'" path)   ; another post
              (let ((slug (match-string 1 path)))
                (unless (member slug (mapcar #'site--slug (site-posts)))
                  (error "%s links to %s.org, which is not a published post"
                         (plist-get info :input-file) slug))
                (org-element-put-property
                 link :path (format "../%s/index.html" slug))))))))))
  tree)

(org-export-define-derived-backend 'site-html 'html
  :translate-alist '((template . site-template))
  :filters-alist '((:filter-parse-tree . site--fix-links)))

;;; Posts

(defun site--slug (file) (file-name-base file))

(defun site--keywords (file)
  "Alist of the #+KEYWORD lines at the top of FILE."
  (with-temp-buffer
    (insert-file-contents file)
    (delay-mode-hooks (org-mode))
    (mapcar (lambda (kw) (cons (car kw) (string-join (cdr kw) " ")))
            (org-collect-keywords
             '("TITLE" "DATE" "FILETAGS" "KIND" "SUMMARY" "DESCRIPTION")))))

(defun site--published-p (file)
  (member "published"
          (split-string (or (cdr (assoc "FILETAGS" (site--keywords file))) "")
                        ":" t "[ \t]+")))

(defvar site--posts nil "Cache for the function `site-posts'.")

(defun site-posts ()
  "The posts to build, newest first."
  (or site--posts
      (setq site--posts
            (sort (seq-filter (lambda (f) (or site-include-drafts (site--published-p f)))
                              (directory-files site-posts t "\\`[^.#].*\\.org\\'"))
                  (lambda (a b)
                    (let ((ka (site--keywords a)) (kb (site--keywords b)))
                      (let ((da (or (cdr (assoc "DATE" ka)) ""))
                            (db (or (cdr (assoc "DATE" kb)) "")))
                        (if (string= da db)
                            (string< (cdr (assoc "TITLE" ka)) (cdr (assoc "TITLE" kb)))
                          (string> da db)))))))))

(defun site--build-post (file)
  (let* ((slug (site--slug file))
         (out (expand-file-name (concat slug "/") site-out))
         (assets (expand-file-name slug site-posts)))
    (message "Building post: %s" slug)
    (if (file-directory-p assets)
        (copy-directory assets out nil t t)
      (make-directory out t))
    (with-current-buffer (find-file-noselect file)
      (org-export-to-file 'site-html (expand-file-name "index.html" out))
      (kill-buffer))))

;;; Front page

(defun site--inline-html (org-markup)
  "Export a one-line ORG-MARKUP string to HTML, without the <p> wrapper."
  (let ((html (string-trim (org-export-string-as org-markup 'html t))))
    (if (string-match "\\`<p>\\(\\(?:.\\|\n\\)*\\)</p>\\'" html)
        (string-trim (match-string 1 html))
      html)))

(defun site--card (file)
  (let ((kw (site--keywords file)))
    (format "        <a class=\"card\" href=\"%s/index.html\">
          <p class=\"card__meta\">%s</p>
          <h3 class=\"card__title\">%s</h3>
          <p class=\"card__summary\">%s</p>
        </a>\n"
            (site--slug file)
            (org-html-encode-plain-text (or (cdr (assoc "KIND" kw)) "Notes"))
            (site--inline-html (cdr (assoc "TITLE" kw)))
            (site--inline-html (or (cdr (assoc "SUMMARY" kw))
                                   (cdr (assoc "DESCRIPTION" kw))
                                   "")))))

(defun site--build-front-page ()
  (with-temp-buffer
    (insert-file-contents (expand-file-name "index.html" site-root))
    (unless (re-search-forward "<!-- posts:begin.*-->\n" nil t)
      (error "index.html has no posts:begin marker"))
    (let ((start (point)))
      (unless (re-search-forward "^.*<!-- posts:end -->" nil t)
        (error "index.html has no posts:end marker"))
      (delete-region start (match-beginning 0))
      (goto-char start)
      (insert (mapconcat #'site--card (site-posts) "")))
    (write-region nil nil (expand-file-name "index.html" site-out))))

;;; Entry point

(defun site-build ()
  "Rebuild _site/ from scratch."
  (when (file-directory-p site-out)
    (delete-directory site-out t))
  (make-directory site-out t)
  (dolist (name site-static)
    (let ((src (expand-file-name name site-root)))
      (if (file-directory-p src)
          (copy-directory src (expand-file-name name site-out) nil t t)
        (copy-file src (expand-file-name name site-out)))))
  (mapc #'site--build-post (site-posts))
  (mapc #'delete-file (directory-files-recursively site-out "\\`\\.DS_Store\\'"))
  (site--build-front-page)
  (message "Built %d post(s) into %s" (length (site-posts)) site-out))

;;; build.el ends here
