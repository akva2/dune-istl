# $Id: Makefile.am 1294 2010-09-20 11:05:20Z mblatt $

SUBDIRS = figures

# only build these programs if an MPI-implementation was found
if MPI
  MPIPROGRAMS = indexset poosc08 poosc08_test
endif
# programs just to build when "make check" is used
check_PROGRAMS = $(MPIPROGRAMS)
# list of tests to run (indicestest is special case)
TESTS = $(MPIPROGRAMS)

if BUILD_DOCS
  DOCFILES = communication.pdf
  EXTRA_DIST = communication.pdf
  EXTRAINSTALL = communication.pdf
endif

# setting like in dune-web
CURDIR=doc/istl/comm
BASEDIR=../../..

# install the docs
docdir=$(datadir)/doc/dune-istl/comm

include $(top_srcdir)/am/latex
include $(top_srcdir)/am/webstuff
include $(top_srcdir)/am/global-rules

poosc08_SOURCES = poosc08.cc
poosc08_CPPFLAGS = $(AM_CPPFLAGS)              \
        $(DUNEMPICPPFLAGS)
poosc08_LDFLAGS = $(AM_LDFLAGS)                \
        $(DUNEMPILDFLAGS)
poosc08_LDADD =                                \
        $(DUNEMPILIBS)                          \
        $(LDADD)

poosc08_test_SOURCES = poosc08_test.cc
poosc08_test_CPPFLAGS = $(AM_CPPFLAGS)              \
        $(DUNEMPICPPFLAGS)
poosc08_test_LDFLAGS = $(AM_LDFLAGS)                \
        $(DUNEMPILDFLAGS)
poosc08_test_LDADD =                                \
        $(DUNEMPILIBS)                          \
        $(LDADD)

indexset_SOURCES = indexset.cc buildindexset.hh reverse.hh
indexset_CPPFLAGS = $(AM_CPPFLAGS)		\
	$(DUNEMPICPPFLAGS)
indexset_LDFLAGS = $(AM_LDFLAGS)		\
	$(DUNEMPILDFLAGS)
indexset_LDADD =				\
	$(DUNEMPILIBS)				\
	$(LDADD)
