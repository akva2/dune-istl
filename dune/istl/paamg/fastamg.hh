// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_ISTL_FASTAMG_HH
#define DUNE_ISTL_FASTAMG_HH

#include <memory>
#include <dune/common/exceptions.hh>
#include <dune/istl/paamg/smoother.hh>
#include <dune/istl/paamg/transfer.hh>
#include <dune/istl/paamg/hierarchy.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/istl/superlu.hh>
#include <dune/istl/solvertype.hh>
#include <dune/istl/io.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/exceptions.hh>

#include "fastamgsmoother.hh"

/** @file
 * @author Markus Blatt
 * @brief A fast AMG method, that currently only allows only Gauss-Seidel
 * smoothing and is currently purely sequential. It
 * combines one Gauss-Seidel presmoothing sweep with
 * the defect calculation to reduce memory transfers.
 */

namespace Dune
{
  namespace Amg
  {
    /**
     * @defgroup ISTL_FSAMG Fast (sequential) Algebraic Multigrid
     * @ingroup ISTL_Prec
     * @brief An Algebraic Multigrid based on Agglomeration that saves memory bandwidth.
     */

    /**
     * @addtogroup ISTL_FSAMG
     *
     * @{
     */

    /**
     * @brief A fast (sequential) algebraic multigrid based on agglomeration that saves memory bandwidth.
     *
     * It combines one Gauss-Seidel smoothing sweep with
     * the defect calculation to reduce memory transfers.
     * \tparam M The matrix type
     * \tparam X The vector type
     * \tparam S Dummy smoother param to be compatible with the normal AMG
     * \tparam PI Currently ignored.
     * \tparam A An allocator for X
     */
    template<class M, class X, class S=double, class PI=SequentialInformation, class A=std::allocator<X> >
    class FastAMG : public Preconditioner<X,X>
    {
    public:
      /** @brief The matrix operator type. */
      typedef M Operator;
      /**
       * @brief The type of the parallel information.
       * Either OwnerOverlapCommunication or another type
       * describing the parallel data distribution and
       * providing communication methods.
       */
      typedef PI ParallelInformation;
      /** @brief The operator hierarchy type. */
      typedef MatrixHierarchy<M, ParallelInformation, A> OperatorHierarchy;
      /** @brief The parallal data distribution hierarchy type. */
      typedef typename OperatorHierarchy::ParallelInformationHierarchy ParallelInformationHierarchy;

      /** @brief The domain type. */
      typedef X Domain;
      /** @brief The range type. */
      typedef X Range;
      /** @brief the type of the coarse solver. */
      typedef InverseOperator<X,X> CoarseSolver;

      typedef typename SmootherTraits<SeqSSOR<typename M::matrix_type,X,X> >::Arguments SmootherArgs;

      enum {
        /** @brief The solver category. */
        category = SolverCategory::sequential
      };

      /**
       * @brief Construct a new amg with a specific coarse solver.
       * @param matrices The already set up matix hierarchy.
       * @param coarseSolver The set up solver to use on the coarse
       * grid, must match the coarse matrix in the matrix hierachy.
       * @param parms The parameters for the AMG.
       */
      FastAMG(const OperatorHierarchy& matrices, CoarseSolver& coarseSolver,
              const Parameters& parms,
              bool symmetric=true);

      /**
       * @brief Construct an AMG with an inexact coarse solver based on the smoother.
       *
       * As coarse solver a preconditioned CG method with the smoother as preconditioner
       * will be used. The matrix hierarchy is built automatically.
       * @param fineOperator The operator on the fine level.
       * @param criterion The criterion describing the coarsening strategy. E. g. SymmetricCriterion
       * or UnsymmetricCriterion, and providing the parameters.
       * @param parms The parameters for the AMG.
       * @param pinfo The information about the parallel distribution of the data.
       */
      template<class C>
      FastAMG(const Operator& fineOperator, const C& criterion,
              const SmootherArgs& smootherArgs,
              const Parameters& parms=Parameters(),
              bool symmetric=true,
              const ParallelInformation& pinfo=ParallelInformation());

      /**
       * @brief Copy constructor.
       */
      FastAMG(const FastAMG& amg),

      ~FastAMG();

      /** \copydoc Preconditioner::pre */
      void pre(Domain& x, Range& b);

      /** \copydoc Preconditioner::apply */
      void apply(Domain& v, const Range& d);

      /** \copydoc Preconditioner::post */
      void post(Domain& x);

      /**
       * @brief Get the aggregate number of each unknown on the coarsest level.
       * @param cont The random access container to store the numbers in.
       */
      template<class A1>
      void getCoarsestAggregateNumbers(std::vector<std::size_t,A1>& cont);

      std::size_t levels();

      std::size_t maxlevels();

      /**
       * @brief Recalculate the matrix hierarchy.
       *
       * It is assumed that the coarsening for the changed fine level
       * matrix would yield the same aggregates. In this case it suffices
       * to recalculate all the Galerkin products for the matrices of the
       * coarser levels.
       */
      void recalculateHierarchy()
      {
        matrices_->recalculateGalerkin(NegateSet<typename PI::OwnerSet>());
      }

      /**
       * @brief Check whether the coarse solver used is a direct solver.
       * @return True if the coarse level solver is a direct solver.
       */
      bool usesDirectCoarseLevelSolver() const;

    private:
      /**
       * @brief Create matrix and smoother hierarchies.
       * @param criterion The coarsening criterion.
       * @param matrix The fine level matrix operator.
       * @param pinfo The fine level parallel information.
       */
      template<class C>
      void createHierarchies(C& criterion, Operator& matrix,
                             const PI& pinfo);

      /** @brief Multigrid cycle on a level. */
      void mgc(Domain& x, const Range& b);

      typename OperatorHierarchy::ParallelMatrixHierarchy::ConstIterator matrix;
      typename ParallelInformationHierarchy::Iterator pinfo;
      typename OperatorHierarchy::RedistributeInfoList::const_iterator redist;
      typename OperatorHierarchy::AggregatesMapList::const_iterator aggregates;
      typename Hierarchy<Domain,A>::Iterator lhs;
      typename Hierarchy<Domain,A>::Iterator residual;
      typename Hierarchy<Range,A>::Iterator rhs;


      /** @brief Apply pre smoothing on the current level. */
      void presmooth(Domain& x, const Range& b);

      /** @brief Apply post smoothing on the current level. */
      void postsmooth(Domain& x, const Range& b);

      /**
       * @brief Move the iterators to the finer level
       * @*/
      void moveToFineLevel(bool processedFineLevel, Domain& coarseX);

      /** @brief Move the iterators to the coarser level */
      bool moveToCoarseLevel();

      /** @brief Initialize iterators over levels with fine level */
      void initIteratorsWithFineLevel();

      /**  @brief The matrix we solve. */
      Dune::shared_ptr<OperatorHierarchy> matrices_;
      /** @brief The solver of the coarsest level. */
      Dune::shared_ptr<CoarseSolver> solver_;
      /** @brief The right hand side of our problem. */
      Hierarchy<Range,A>* rhs_;
      /** @brief The left approximate solution of our problem. */
      Hierarchy<Domain,A>* lhs_;
      /** @brief The current residual. */
      Hierarchy<Domain,A>* residual_;

      /** @brief The type of the chooser of the scalar product. */
      typedef ScalarProductChooser<X,PI,M::category> ScalarProductChooserType;
      /** @brief The type of the scalar product for the coarse solver. */
      typedef typename ScalarProductChooserType::ScalarProduct ScalarProduct;
      typedef Dune::shared_ptr<ScalarProduct> ScalarProductPointer;
      /** @brief Scalar product on the coarse level. */
      ScalarProductPointer scalarProduct_;
      /** @brief Gamma, 1 for V-cycle and 2 for W-cycle. */
      std::size_t gamma_;
      /** @brief The number of pre and postsmoothing steps. */
      std::size_t preSteps_;
      /** @brief The number of postsmoothing steps. */
      std::size_t postSteps_;
      std::size_t level;
      bool buildHierarchy_;
      bool symmetric;
      bool coarsesolverconverged;
      typedef SeqSSOR<typename M::matrix_type,X,X> Smoother;
      typedef Dune::shared_ptr<Smoother> SmootherPointer;
      SmootherPointer coarseSmoother_;
      /** @brief The verbosity level. */
      std::size_t verbosity_;
    };

    template<class M, class X, class S, class PI, class A>
    FastAMG<M,X,S,PI,A>::FastAMG(const FastAMG& amg)
    : matrices_(amg.matrices_), solver_(amg.solver_),
      rhs_(), lhs_(), residual_(), scalarProduct_(amg.scalarProduct_),
      gamma_(amg.gamma_), preSteps_(amg.preSteps_), postSteps_(amg.postSteps_),
      symmetric(amg.symmetric), coarsesolverconverged(amg.coarsesolverconverged),
      coarseSmoother_(amg.coarseSmoother_), verbosity_(amg.verbosity_),
      buildHierarchy_(false)
    {
      if(amg.rhs_)
        rhs_=new Hierarchy<Range,A>(*amg.rhs_);
      if(amg.lhs_)
        lhs_=new Hierarchy<Domain,A>(*amg.lhs_);
      if(amg.residual_)
        residual_=new Hierarchy<Domain,A>(*amg.residual_);
    }

    template<class M, class X, class S, class PI, class A>
    FastAMG<M,X,S,PI,A>::FastAMG(const OperatorHierarchy& matrices, CoarseSolver& coarseSolver,
                                 const Parameters& parms, bool symmetric_)
      : matrices_(&matrices), solver_(&coarseSolver),
        rhs_(), lhs_(), residual_(), scalarProduct_(),
        gamma_(parms.getGamma()), preSteps_(parms.getNoPreSmoothSteps()),
        postSteps_(parms.getNoPostSmoothSteps()), buildHierarchy_(false),
        symmetric(symmetric_), coarsesolverconverged(true),
        coarseSmoother_(), verbosity_(parms.debugLevel())
    {
      if(preSteps_>1||postSteps_>1)
      {
        std::cerr<<"WARNING only one step of smoothing is supported!"<<std::endl;
        preSteps_=postSteps_=0;
      }
      assert(matrices_->isBuilt());
      dune_static_assert((is_same<PI,SequentialInformation>::value), "Currently only sequential runs are supported");
    }
    template<class M, class X, class S, class PI, class A>
    template<class C>
    FastAMG<M,X,S,PI,A>::FastAMG(const Operator& matrix,
                               const C& criterion,
                               const SmootherArgs& smootherArgs,
                               const Parameters& parms,
                               bool symmetric_,
                               const PI& pinfo)
      : solver_(), rhs_(), lhs_(), residual_(), scalarProduct_(), gamma_(parms.getGamma()),
        preSteps_(parms.getNoPreSmoothSteps()), postSteps_(parms.getNoPostSmoothSteps()),
        buildHierarchy_(true),
        symmetric(symmetric_), coarsesolverconverged(true),
        coarseSmoother_(), verbosity_(criterion.debugLevel())
    {
      if(preSteps_>1||postSteps_>1)
      {
        std::cerr<<"WARNING only one step of smoothing is supported!"<<std::endl;
        preSteps_=postSteps_=1;
      }
      dune_static_assert((is_same<PI,SequentialInformation>::value), "Currently only sequential runs are supported");
      // TODO: reestablish compile time checks.
      //dune_static_assert(static_cast<int>(PI::category)==static_cast<int>(S::category),
      //			 "Matrix and Solver must match in terms of category!");
      createHierarchies(criterion, const_cast<Operator&>(matrix), pinfo);
    }

    template<class M, class X, class S, class PI, class A>
    FastAMG<M,X,S,PI,A>::~FastAMG()
    {
      if(buildHierarchy_) {
        if(solver_)
          solver_.reset();
        if(coarseSmoother_)
          coarseSmoother_.reset();
      }
      if(lhs_)
        delete lhs_;
      lhs_=nullptr;
      if(residual_)
        delete residual_;
      residual_=nullptr;
      if(rhs_)
        delete rhs_;
      rhs_=nullptr;
    }

    template<class M, class X, class S, class PI, class A>
    template<class C>
    void FastAMG<M,X,S,PI,A>::createHierarchies(C& criterion, Operator& matrix,
                                                const PI& pinfo)
    {
      Timer watch;
      matrices_.reset(new OperatorHierarchy(matrix, pinfo));

      matrices_->template build<NegateSet<typename PI::OwnerSet> >(criterion);

      if(verbosity_>0 && matrices_->parallelInformation().finest()->communicator().rank()==0)
        std::cout<<"Building Hierarchy of "<<matrices_->maxlevels()<<" levels took "<<watch.elapsed()<<" seconds."<<std::endl;

      if(buildHierarchy_ && matrices_->levels()==matrices_->maxlevels()) {
        // We have the carsest level. Create the coarse Solver
        typedef typename SmootherTraits<Smoother>::Arguments SmootherArgs;
        SmootherArgs sargs;
        sargs.iterations = 1;

        typename ConstructionTraits<Smoother>::Arguments cargs;
        cargs.setArgs(sargs);
        if(matrices_->redistributeInformation().back().isSetup()) {
          // Solve on the redistributed partitioning
          cargs.setMatrix(matrices_->matrices().coarsest().getRedistributed().getmat());
          cargs.setComm(matrices_->parallelInformation().coarsest().getRedistributed());
        }else{
          cargs.setMatrix(matrices_->matrices().coarsest()->getmat());
          cargs.setComm(*matrices_->parallelInformation().coarsest());
        }

        coarseSmoother_.reset(ConstructionTraits<Smoother>::construct(cargs));
        scalarProduct_.reset(ScalarProductChooserType::construct(cargs.getComm()));

#if HAVE_SUPERLU
        // Use superlu if we are purely sequential or with only one processor on the coarsest level.
        if(is_same<ParallelInformation,SequentialInformation>::value // sequential mode
           || matrices_->parallelInformation().coarsest()->communicator().size()==1 //parallel mode and only one processor
           || (matrices_->parallelInformation().coarsest().isRedistributed()
               && matrices_->parallelInformation().coarsest().getRedistributed().communicator().size()==1
               && matrices_->parallelInformation().coarsest().getRedistributed().communicator().size()>0)) { // redistribute and 1 proc
          if(verbosity_>0 && matrices_->parallelInformation().coarsest()->communicator().rank()==0)
            std::cout<<"Using superlu"<<std::endl;
          if(matrices_->parallelInformation().coarsest().isRedistributed())
          {
            if(matrices_->matrices().coarsest().getRedistributed().getmat().N()>0)
              // We are still participating on this level
              solver_.reset(new SuperLU<typename M::matrix_type>(matrices_->matrices().coarsest().getRedistributed().getmat(), false, false));
            else
              solver_.reset();
          }else
            solver_.reset(new SuperLU<typename M::matrix_type>(matrices_->matrices().coarsest()->getmat(), false, false));
        }else
#endif
        {
          if(matrices_->parallelInformation().coarsest().isRedistributed())
          {
            if(matrices_->matrices().coarsest().getRedistributed().getmat().N()>0)
              // We are still participating on this level
              solver_.reset(new BiCGSTABSolver<X>(const_cast<M&>(matrices_->matrices().coarsest().getRedistributed()),
                                                  *scalarProduct_,
                                                  *coarseSmoother_, 1E-2, 1000, 0));
            else
              solver_.reset();
          }else
            solver_.reset(new BiCGSTABSolver<X>(const_cast<M&>(*matrices_->matrices().coarsest()),
                                                *scalarProduct_,
                                                *coarseSmoother_, 1E-2, 1000, 0));
        }
      }

      if(verbosity_>0 && matrices_->parallelInformation().finest()->communicator().rank()==0)
        std::cout<<"Building Hierarchy of "<<matrices_->maxlevels()<<" levels took "<<watch.elapsed()<<" seconds."<<std::endl;
    }


    template<class M, class X, class S, class PI, class A>
    void FastAMG<M,X,S,PI,A>::pre(Domain& x, Range& b)
    {
      Timer watch, watch1;
      // Detect Matrix rows where all offdiagonal entries are
      // zero and set x such that  A_dd*x_d=b_d
      // Thus users can be more careless when setting up their linear
      // systems.
      typedef typename M::matrix_type Matrix;
      typedef typename Matrix::ConstRowIterator RowIter;
      typedef typename Matrix::ConstColIterator ColIter;
      typedef typename Matrix::block_type Block;
      Block zero;
      zero=typename Matrix::field_type();

      const Matrix& mat=matrices_->matrices().finest()->getmat();
      for(RowIter row=mat.begin(); row!=mat.end(); ++row) {
        bool isDirichlet = true;
        bool hasDiagonal = false;
        ColIter diag;
        for(ColIter col=row->begin(); col!=row->end(); ++col) {
          if(row.index()==col.index()) {
            diag = col;
            hasDiagonal = false;
          }else{
            if(*col!=zero)
              isDirichlet = false;
          }
        }
        if(isDirichlet && hasDiagonal)
          diag->solve(x[row.index()], b[row.index()]);
      }
      std::cout<<" Preprocessing Dirichlet took "<<watch1.elapsed()<<std::endl;
      watch1.reset();
      // No smoother to make x consistent! Do it by hand
      matrices_->parallelInformation().coarsest()->copyOwnerToAll(x,x);
      Range* copy = new Range(b);
      if(rhs_)
        delete rhs_;
      rhs_ = new Hierarchy<Range,A>(copy);
      Domain* dcopy = new Domain(x);
      if(lhs_)
        delete lhs_;
      lhs_ = new Hierarchy<Domain,A>(dcopy);
      dcopy = new Domain(x);
      residual_ = new Hierarchy<Domain,A>(dcopy);
      matrices_->coarsenVector(*rhs_);
      matrices_->coarsenVector(*lhs_);
      matrices_->coarsenVector(*residual_);

      // The preconditioner might change x and b. So we have to
      // copy the changes to the original vectors.
      x = *lhs_->finest();
      b = *rhs_->finest();
    }
    template<class M, class X, class S, class PI, class A>
    std::size_t FastAMG<M,X,S,PI,A>::levels()
    {
      return matrices_->levels();
    }
    template<class M, class X, class S, class PI, class A>
    std::size_t FastAMG<M,X,S,PI,A>::maxlevels()
    {
      return matrices_->maxlevels();
    }

    /** \copydoc Preconditioner::apply */
    template<class M, class X, class S, class PI, class A>
    void FastAMG<M,X,S,PI,A>::apply(Domain& v, const Range& d)
    {
      // Init all iterators for the current level
      initIteratorsWithFineLevel();

      assert(v.two_norm()==0);

      level=0;
      if(matrices_->maxlevels()==1){
        // The coarse solver might modify the d!
        Range b(d);
        mgc(v, b);
      }else
        mgc(v, d);
      if(postSteps_==0||matrices_->maxlevels()==1)
        pinfo->copyOwnerToAll(v, v);
    }

    template<class M, class X, class S, class PI, class A>
    void FastAMG<M,X,S,PI,A>::initIteratorsWithFineLevel()
    {
      matrix = matrices_->matrices().finest();
      pinfo = matrices_->parallelInformation().finest();
      redist =
        matrices_->redistributeInformation().begin();
      aggregates = matrices_->aggregatesMaps().begin();
      lhs = lhs_->finest();
      residual = residual_->finest();
      rhs = rhs_->finest();
    }

    template<class M, class X, class S, class PI, class A>
    bool FastAMG<M,X,S,PI,A>
    ::moveToCoarseLevel()
    {
      bool processNextLevel=true;

      if(redist->isSetup()) {
        throw "bla";
        redist->redistribute(static_cast<const Range&>(*residual), residual.getRedistributed());
        processNextLevel =residual.getRedistributed().size()>0;
        if(processNextLevel) {
          //restrict defect to coarse level right hand side.
          ++pinfo;
          Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
          ::restrictVector(*(*aggregates), *rhs, static_cast<const Range&>(residual.getRedistributed()), *pinfo);
        }
      }else{
        //restrict defect to coarse level right hand side.
        typename Hierarchy<Range,A>::Iterator fineRhs = rhs++;
        ++pinfo;
        Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
        ::restrictVector(*(*aggregates), *rhs, static_cast<const Range&>(*residual), *pinfo);
      }

      if(processNextLevel) {
        // prepare coarse system
        ++residual;
        ++lhs;
        ++matrix;
        ++level;
        ++redist;

        if(matrix != matrices_->matrices().coarsest() || matrices_->levels()<matrices_->maxlevels()) {
          // next level is not the globally coarsest one
          ++aggregates;
        }
        // prepare the lhs on the next level
        *lhs=0;
        *residual=0;
      }
      return processNextLevel;
    }

    template<class M, class X, class S, class PI, class A>
    void FastAMG<M,X,S,PI,A>
    ::moveToFineLevel(bool processNextLevel, Domain& x)
    {
      if(processNextLevel) {
        if(matrix != matrices_->matrices().coarsest() || matrices_->levels()<matrices_->maxlevels()) {
          // previous level is not the globally coarsest one
          --aggregates;
        }
        --redist;
        --level;
        //prolongate and add the correction (update is in coarse left hand side)
        --matrix;
        --residual;

      }

      typename Hierarchy<Domain,A>::Iterator coarseLhs = lhs--;
      if(redist->isSetup()) {

        // Need to redistribute during prolongate
        Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
        ::prolongateVector(*(*aggregates), *coarseLhs, x, lhs.getRedistributed(), matrices_->getProlongationDampingFactor(),
                           *pinfo, *redist);
      }else{
        Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
        ::prolongateVector(*(*aggregates), *coarseLhs, x,
                           matrices_->getProlongationDampingFactor(), *pinfo);

        // printvector(std::cout, *lhs, "prolongated coarse grid correction", "lhs", 10, 10, 10);
      }


      if(processNextLevel) {
        --rhs;
      }

    }


    template<class M, class X, class S, class PI, class A>
    void FastAMG<M,X,S,PI,A>
    ::presmooth(Domain& x, const Range& b)
    {
      GaussSeidelPresmoothDefect<M::matrix_type::blocklevel>::apply(matrix->getmat(),
                                                                    x,
                                                                    *residual,
                                                                    b);
    }

    template<class M, class X, class S, class PI, class A>
    void FastAMG<M,X,S,PI,A>
    ::postsmooth(Domain& x, const Range& b)
    {
      GaussSeidelPostsmoothDefect<M::matrix_type::blocklevel>
      ::apply(matrix->getmat(), x, *residual, b);
    }


    template<class M, class X, class S, class PI, class A>
    bool FastAMG<M,X,S,PI,A>::usesDirectCoarseLevelSolver() const
    {
      return IsDirectSolver< CoarseSolver>::value;
    }

    template<class M, class X, class S, class PI, class A>
    void FastAMG<M,X,S,PI,A>::mgc(Domain& v, const Range& b){

      if(matrix == matrices_->matrices().coarsest() && levels()==maxlevels()) {
        // Solve directly
        InverseOperatorResult res;
        res.converged=true; // If we do not compute this flag will not get updated
        if(redist->isSetup()) {
          redist->redistribute(b, rhs.getRedistributed());
          if(rhs.getRedistributed().size()>0) {
            // We are still participating in the computation
            pinfo.getRedistributed().copyOwnerToAll(rhs.getRedistributed(), rhs.getRedistributed());
            solver_->apply(lhs.getRedistributed(), rhs.getRedistributed(), res);
          }
          redist->redistributeBackward(v, lhs.getRedistributed());
          pinfo->copyOwnerToAll(v, v);
        }else{
          pinfo->copyOwnerToAll(b, b);
          solver_->apply(v, const_cast<Range&>(b), res);
        }

        // printvector(std::cout, *lhs, "coarse level update", "u", 10, 10, 10);
        // printvector(std::cout, *rhs, "coarse level rhs", "rhs", 10, 10, 10);
        if (!res.converged)
          coarsesolverconverged = false;
      }else{
        // presmoothing
        presmooth(v, b);
        // printvector(std::cout, *lhs, "update", "u", 10, 10, 10);
        // printvector(std::cout, *residual, "post presmooth residual", "r", 10);
#ifndef DUNE_AMG_NO_COARSEGRIDCORRECTION
        bool processNextLevel = moveToCoarseLevel();

        if(processNextLevel) {
          // next level
          for(std::size_t i=0; i<gamma_; i++)
            mgc(*lhs, *rhs);
        }

        moveToFineLevel(processNextLevel, v);
#else
        *lhs=0;
#endif

        if(matrix == matrices_->matrices().finest()) {
          coarsesolverconverged = matrices_->parallelInformation().finest()->communicator().prod(coarsesolverconverged);
          if(!coarsesolverconverged)
            DUNE_THROW(MathError, "Coarse solver did not converge");
        }

        // printvector(std::cout, *lhs, "update corrected", "u", 10, 10, 10);
        // postsmoothing
        postsmooth(v, b);
        // printvector(std::cout, *lhs, "update postsmoothed", "u", 10, 10, 10);

      }
    }


    /** \copydoc Preconditioner::post */
    template<class M, class X, class S, class PI, class A>
    void FastAMG<M,X,S,PI,A>::post(Domain& x)
    {
      delete lhs_;
      lhs_=nullptr;
      delete rhs_;
      rhs_=nullptr;
      delete residual_;
      residual_=nullptr;
    }

    template<class M, class X, class S, class PI, class A>
    template<class A1>
    void FastAMG<M,X,S,PI,A>::getCoarsestAggregateNumbers(std::vector<std::size_t,A1>& cont)
    {
      matrices_->getCoarsestAggregatesOnFinest(cont);
    }

  } // end namespace Amg
} // end namespace Dune

#endif
