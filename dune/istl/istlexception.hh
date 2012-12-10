#ifndef DUNE_ISTLEXC_HH
#define DUNE_ISTLEXC_HH

#include <dune/common/exceptions.hh>

namespace Dune {
   
    /** 
		@addtogroup ISTL
		@{
     */

  //! derive error class from the base class in common
  class ISTLError : public Dune::MathError {};

  /** @} end documentation */

} // end namespace

#endif
