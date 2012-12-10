# $Id: Makefile.am 1408 2010-11-29 12:30:48Z mblatt $

dist_noinst_DATA = example.cc

noinst_PROGRAMS = example
example_SOURCES = example.cc
example_CPPFLAGS = $(AM_CPPFLAGS) $(BOOST_CPPFLAGS) $(DUNE_CPPFLAGS)

include $(top_srcdir)/am/global-rules
