// -*- tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=8 sw=4 sts=4:
#ifndef DUNE_ISTL_SOLVERTYPE_HH
#define DUNE_ISTL_SOLVERTYPE_HH

/**
 * @file
 * @brief Templates characterizing the type of a solver.
 */
namespace Dune
{
    template<typename Solver>
    struct IsDirectSolver
    {
        enum
        {
            /** 
             * @brief Whether this is a direct solver.
             *
             * If Solver is a direct solver, this is true.
             */
            value =false
        };
    };
}// end namespace Dune
#endif
