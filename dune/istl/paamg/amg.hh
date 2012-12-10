// $Id: amg.hh 1547 2012-04-19 17:19:40Z mblatt $
#ifndef DUNE_AMG_AMG_HH
#define DUNE_AMG_AMG_HH

#include<memory>
#include<dune/common/exceptions.hh>
#include<dune/istl/paamg/smoother.hh>
#include<dune/istl/paamg/transfer.hh>
#include<dune/istl/paamg/hierarchy.hh>
#include<dune/istl/solvers.hh>
#include<dune/istl/scalarproducts.hh>
#include<dune/istl/superlu.hh>
#include<dune/istl/solvertype.hh>
#include<dune/common/typetraits.hh>
#include<dune/common/exceptions.hh>

#include <omp.h>

namespace Dune
{
  namespace Amg
  {
    /**
     * @defgroup ISTL_PAAMG Parallel Algebraic Multigrid
     * @ingroup ISTL_Prec
     * @brief A Parallel Algebraic Multigrid based on Agglomeration.
     */
    
    /**
     * @addtogroup ISTL_PAAMG
     *
     * @{
     */
    
    /** @file
     * @author Markus Blatt
     * @brief The AMG preconditioner.
     */

    template<class M, class X, class S, class P, class K, class A>
    class KAMG;
    
    template<class T>
    class KAmgTwoGrid;
    
    /**
     * @brief Parallel algebraic multigrid based on agglomeration.
     *
     * \tparam M The matrix type
     * \tparam X The vector type
     * \tparam A An allocator for X
     */
    template<class M, class X, class S, class PI=SequentialInformation,
	     class A=std::allocator<X> >
    class AMG : public Preconditioner<X,X>
    {
      template<class M1, class X1, class S1, class P1, class K1, class A1>
      friend class KAMG;
      
      friend class KAmgTwoGrid<AMG>;
      
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
      /** 
       * @brief The type of the smoother. 
       *
       * One of the preconditioners implementing the Preconditioner interface.
       * Note that the smoother has to fit the ParallelInformation.*/
      typedef S Smoother;
  
      /** @brief The argument type for the construction of the smoother. */
      typedef typename SmootherTraits<Smoother>::Arguments SmootherArgs;
      
      enum {
	/** @brief The solver category. */
	category = S::category
      };

      /**
       * @brief Construct a new amg with a specific coarse solver.
       * @param matrices The already set up matix hierarchy.
       * @param coarseSolver The set up solver to use on the coarse 
       * grid, must match the coarse matrix in the matrix hierachy.
       * @param smootherArgs The  arguments needed for thesmoother to use 
       * for pre and post smoothing
       * @param gamma The number of subcycles. 1 for V-cycle, 2 for W-cycle.
       * @param preSmoothingSteps The number of smoothing steps for premoothing.
       * @param postSmoothingSteps The number of smoothing steps for postmoothing.
       * @deprecated Use constructor
       * AMG(const OperatorHierarchy&, CoarseSolver&, const SmootherArgs, const Parameters&)
       * instead.
       * All parameters can be set in the criterion!
       */
      AMG(const OperatorHierarchy& matrices, CoarseSolver& coarseSolver, 
	  const SmootherArgs& smootherArgs, std::size_t gamma,
	  std::size_t preSmoothingSteps,
	  std::size_t postSmoothingSteps, 
          bool additive=false) DUNE_DEPRECATED;

      /**
       * @brief Construct a new amg with a specific coarse solver.
       * @param matrices The already set up matix hierarchy.
       * @param coarseSolver The set up solver to use on the coarse 
       * grid, must match the coarse matrix in the matrix hierachy.
       * @param smootherArgs The  arguments needed for thesmoother to use 
       * for pre and post smoothing.
       * @param parms The parameters for the AMG.
       */
      AMG(const OperatorHierarchy& matrices, CoarseSolver& coarseSolver, 
	  const SmootherArgs& smootherArgs, const Parameters& parms);

      /**
       * @brief Construct an AMG with an inexact coarse solver based on the smoother.
       *
       * As coarse solver a preconditioned CG method with the smoother as preconditioner
       * will be used. The matrix hierarchy is built automatically.
       * @param fineOperator The operator on the fine level.
       * @param criterion The criterion describing the coarsening strategy. E. g. SymmetricCriterion
       * or UnsymmetricCriterion.
       * @param smootherArgs The arguments for constructing the smoothers.
       * @param gamma 1 for V-cycle, 2 for W-cycle
       * @param preSmoothingSteps The number of smoothing steps for premoothing.
       * @param postSmoothingSteps The number of smoothing steps for postmoothing.
       * @param pinfo The information about the parallel distribution of the data.
       * @deprecated Use 
       * AMG(const Operator&, const C&, const SmootherArgs, const ParallelInformation)
       * instead.
       * All parameters can be set in the criterion!
       */
      template<class C>
      AMG(const Operator& fineOperator, const C& criterion,
	  const SmootherArgs& smootherArgs, std::size_t gamma,
	  std::size_t preSmoothingSteps, 
          std::size_t postSmoothingSteps,
	  bool additive=false, 
          const ParallelInformation& pinfo=ParallelInformation()) DUNE_DEPRECATED;

      /**
       * @brief Construct an AMG with an inexact coarse solver based on the smoother.
       *
       * As coarse solver a preconditioned CG method with the smoother as preconditioner
       * will be used. The matrix hierarchy is built automatically.
       * @param fineOperator The operator on the fine level.
       * @param criterion The criterion describing the coarsening strategy. E. g. SymmetricCriterion
       * or UnsymmetricCriterion, and providing the parameters.
       * @param smootherArgs The arguments for constructing the smoothers.
       * @param pinfo The information about the parallel distribution of the data.
       */
      template<class C>
      AMG(const Operator& fineOperator, const C& criterion,
	  const SmootherArgs& smootherArgs,
          const ParallelInformation& pinfo=ParallelInformation());

      ~AMG();

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

      void setContext(const std::string& context)
      {
        size_t entry = omp_get_thread_num();
        threadContext[entry] = context;
      }

      void addContext(const std::string& context)
      {
        size_t entry = omp_get_max_threads();
        sol[context].resize(entry);
        if (threadContext.size() != entry)
          threadContext.resize(entry);
      }
      
    private:
      /** @brief Multigrid cycle on a level. */
      void mgc();

      void additiveMgc();

      /** @brief Apply pre smoothing on the current level. */
      void presmooth();

      /** @brief Apply post smoothing on the current level. */
      void postsmooth();
      
      /** 
       * @brief Move the iterators to the finer level 
       * @*/
      void moveToFineLevel(bool processedFineLevel);

      /** @brief Move the iterators to the coarser level */
      bool moveToCoarseLevel();

      /** @brief Initialize iterators over levels with fine level */
      void initIteratorsWithFineLevel();

      /**  @brief The matrix we solve. */
      OperatorHierarchy* matrices_;
      /** @brief The arguments to construct the smoother */
      SmootherArgs smootherArgs_;
      /** @brief The hierarchy of the smoothers. */
      Hierarchy<Smoother,A> smoothers_;
      /** @brief The solver of the coarsest level. */
      CoarseSolver* solver_;

      bool predone;

      typedef struct ThreadContext {
        /** @brief The right hand side of our problem. */
        Hierarchy<Range,A>* rhs_;
        /** @brief The left approximate solution of our problem. */
        Hierarchy<Domain,A>* lhs_;
        /** @brief The total update for the outer solver. */
        Hierarchy<Domain,A>* update_;
        typename Hierarchy<Smoother,A>::Iterator smoother;
        typename OperatorHierarchy::ParallelMatrixHierarchy::ConstIterator matrix;
        typename ParallelInformationHierarchy::Iterator pinfo;
        typename OperatorHierarchy::RedistributeInfoList::const_iterator redist;
        typename OperatorHierarchy::AggregatesMapList::const_iterator aggregates;
        typename Hierarchy<Domain,A>::Iterator lhs;
        typename Hierarchy<Domain,A>::Iterator update;
        typename Hierarchy<Range,A>::Iterator rhs;
        std::size_t level;
      
        ThreadContext()
        {
          rhs_ = 0;
          lhs_ = 0;
          update_ = 0;
        }
      } ThreadContext;

      //! \brief Maps from solution context to thread context
      std::map<std::string, std::vector<ThreadContext> > sol;
      
      //! \brief Current solution context of each thread
      std::vector<std::string> threadContext;

      //! \brief Returns the context for the current thread
      ThreadContext& getThreadContext()
      {
        size_t entry = omp_get_thread_num();
        return sol[threadContext[entry]][entry];
      }

      /** @brief The type of the chooser of the scalar product. */
      typedef Dune::ScalarProductChooser<X,PI,M::category> ScalarProductChooser;
      /** @brief The type of the scalar product for the coarse solver. */
      typedef typename ScalarProductChooser::ScalarProduct ScalarProduct;
      /** @brief Scalar product on the coarse level. */
      ScalarProduct* scalarProduct_;
      /** @brief Gamma, 1 for V-cycle and 2 for W-cycle. */
      std::size_t gamma_;
      /** @brief The number of pre and postsmoothing steps. */
      std::size_t preSteps_;
      /** @brief The number of postsmoothing steps. */
      std::size_t postSteps_;
      bool buildHierarchy_;
      bool additive;
      bool coarsesolverconverged;
      Smoother *coarseSmoother_;
      /** @brief The verbosity level. */
      std::size_t verbosity_;
    };

    template<class M, class X, class S, class PI, class A>
    AMG<M,X,S,PI,A>::AMG(const OperatorHierarchy& matrices, CoarseSolver& coarseSolver, 
			const SmootherArgs& smootherArgs,
			std::size_t gamma, std::size_t preSmoothingSteps,
			std::size_t postSmoothingSteps, bool additive_)
      : matrices_(&matrices), smootherArgs_(smootherArgs),
	smoothers_(), solver_(&coarseSolver), scalarProduct_(0),
	gamma_(gamma), preSteps_(preSmoothingSteps), postSteps_(postSmoothingSteps), buildHierarchy_(false),
	additive(additive_), coarsesolverconverged(true),
	coarseSmoother_(), verbosity_(2)
    {
      assert(matrices_->isBuilt());
      
      // build the necessary smoother hierarchies
      matrices_->coarsenSmoother(smoothers_, smootherArgs_);
    }

    template<class M, class X, class S, class PI, class A>
    AMG<M,X,S,PI,A>::AMG(const OperatorHierarchy& matrices, CoarseSolver& coarseSolver, 
                         const SmootherArgs& smootherArgs,
                         const Parameters& parms)
      : matrices_(&matrices), smootherArgs_(smootherArgs),
	smoothers_(), solver_(&coarseSolver), scalarProduct_(0),
	gamma_(parms.getGamma()), preSteps_(parms.getNoPreSmoothSteps()), 
        postSteps_(parms.getNoPostSmoothSteps()), buildHierarchy_(false),
	additive(parms.getAdditive()), coarsesolverconverged(true),
	coarseSmoother_(), verbosity_(parms.debugLevel())
    {
      assert(matrices_->isBuilt());
      
      // build the necessary smoother hierarchies
      matrices_->coarsenSmoother(smoothers_, smootherArgs_);
    }

    template<class M, class X, class S, class PI, class A>
    template<class C>
    AMG<M,X,S,PI,A>::AMG(const Operator& matrix,
			const C& criterion,
			const SmootherArgs& smootherArgs,
			std::size_t gamma, std::size_t preSmoothingSteps,
			std::size_t postSmoothingSteps,
			bool additive_,
			const PI& pinfo)
      : smootherArgs_(smootherArgs),
	smoothers_(), solver_(), scalarProduct_(0), gamma_(gamma),
	preSteps_(preSmoothingSteps), postSteps_(postSmoothingSteps), buildHierarchy_(true),
	additive(additive_), coarsesolverconverged(true),
	coarseSmoother_(), verbosity_(criterion.debugLevel())
    {
      dune_static_assert(static_cast<int>(M::category)==static_cast<int>(S::category),
			 "Matrix and Solver must match in terms of category!");
      // TODO: reestablish compile time checks.
      //dune_static_assert(static_cast<int>(PI::category)==static_cast<int>(S::category),
      //			 "Matrix and Solver must match in terms of category!");
      Timer watch;
      matrices_ = new OperatorHierarchy(const_cast<Operator&>(matrix), pinfo);
            
      matrices_->template build<NegateSet<typename PI::OwnerSet> >(criterion);
      
      // build the necessary smoother hierarchies
      matrices_->coarsenSmoother(smoothers_, smootherArgs_);

      if(verbosity_>0 && matrices_->parallelInformation().finest()->communicator().rank()==0)
	std::cout<<"Building Hierarchy of "<<matrices_->maxlevels()<<" levels took "<<watch.elapsed()<<" seconds."<<std::endl;
    }

    template<class M, class X, class S, class PI, class A>
    template<class C>
    AMG<M,X,S,PI,A>::AMG(const Operator& matrix,
			const C& criterion,
                         const SmootherArgs& smootherArgs,
			const PI& pinfo)
      : smootherArgs_(smootherArgs),
	smoothers_(), solver_(), scalarProduct_(0), 
        gamma_(criterion.getGamma()), preSteps_(criterion.getNoPreSmoothSteps()), 
        postSteps_(criterion.getNoPostSmoothSteps()), buildHierarchy_(true),
	additive(criterion.getAdditive()), coarsesolverconverged(true),
	coarseSmoother_(), verbosity_(criterion.debugLevel())
    {
      dune_static_assert(static_cast<int>(M::category)==static_cast<int>(S::category),
			 "Matrix and Solver must match in terms of category!");
      // TODO: reestablish compile time checks.
      //dune_static_assert(static_cast<int>(PI::category)==static_cast<int>(S::category),
      //			 "Matrix and Solver must match in terms of category!");
      Timer watch;
      matrices_ = new OperatorHierarchy(const_cast<Operator&>(matrix), pinfo);
            
      matrices_->template build<NegateSet<typename PI::OwnerSet> >(criterion);
      
      // build the necessary smoother hierarchies
      matrices_->coarsenSmoother(smoothers_, smootherArgs_);

      if(verbosity_>0 && matrices_->parallelInformation().finest()->communicator().rank()==0)
	std::cout<<"Building Hierarchy of "<<matrices_->maxlevels()<<" levels took "<<watch.elapsed()<<" seconds."<<std::endl;
      predone = false;
    }
    
    template<class M, class X, class S, class PI, class A>
    AMG<M,X,S,PI,A>::~AMG()
    {
      if(buildHierarchy_){
	delete matrices_;
      }
      if(scalarProduct_)
	delete scalarProduct_;
    }

    
    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::pre(Domain& x, Range& b)
    {
      ThreadContext& context = getThreadContext();

      if(smoothers_.levels()>0)
	smoothers_.finest()->pre(x,b);
      else
	// No smoother to make x consistent! Do it by hand
	matrices_->parallelInformation().coarsest()->copyOwnerToAll(x,x);
      Range* copy = new Range(b);
      context.rhs_ = new Hierarchy<Range,A>(*copy);
      Domain* dcopy = new Domain(x);
      context.lhs_ = new Hierarchy<Domain,A>(*dcopy);
      dcopy = new Domain(x);
      context.update_ = new Hierarchy<Domain,A>(*dcopy);
      matrices_->coarsenVector(*context.rhs_);
      matrices_->coarsenVector(*context.lhs_);
      matrices_->coarsenVector(*context.update_);
      
      // Preprocess all smoothers
      typedef typename Hierarchy<Smoother,A>::Iterator Iterator;
      typedef typename Hierarchy<Range,A>::Iterator RIterator;
      typedef typename Hierarchy<Domain,A>::Iterator DIterator;
      Iterator coarsest = smoothers_.coarsest();
      Iterator smoother = smoothers_.finest();
      RIterator rhs = context.rhs_->finest();
      DIterator lhs = context.lhs_->finest();
      if(smoothers_.levels()>0){
	  
      assert(context.lhs_->levels()==context.rhs_->levels());
      assert(smoothers_.levels()==context.lhs_->levels() || matrices_->levels()==matrices_->maxlevels());
      assert(smoothers_.levels()+1==context.lhs_->levels() || matrices_->levels()<matrices_->maxlevels());
      
      if(smoother!=coarsest)
	for(++smoother, ++lhs, ++rhs; smoother != coarsest; ++smoother, ++lhs, ++rhs)
	  smoother->pre(*lhs,*rhs);
      smoother->pre(*lhs,*rhs);
      }
      
      
      // The preconditioner might change x and b. So we have to 
      // copy the changes to the original vectors.
      x = *context.lhs_->finest();
      b = *context.rhs_->finest();
      
      if(!predone && buildHierarchy_ && matrices_->levels()==matrices_->maxlevels()){
	// We have the carsest level. Create the coarse Solver
	SmootherArgs sargs(smootherArgs_);
	sargs.iterations = 1;
	
	typename ConstructionTraits<Smoother>::Arguments cargs;
	cargs.setArgs(sargs);
	if(matrices_->redistributeInformation().back().isSetup()){
	  // Solve on the redistributed partitioning     
	  cargs.setMatrix(matrices_->matrices().coarsest().getRedistributed().getmat());
	  cargs.setComm(matrices_->parallelInformation().coarsest().getRedistributed());
	  
	  coarseSmoother_ = ConstructionTraits<Smoother>::construct(cargs);
	  scalarProduct_ = ScalarProductChooser::construct(matrices_->parallelInformation().coarsest().getRedistributed());
	}else{    
	  cargs.setMatrix(matrices_->matrices().coarsest()->getmat());
	  cargs.setComm(*matrices_->parallelInformation().coarsest());
	  
	  coarseSmoother_ = ConstructionTraits<Smoother>::construct(cargs);
	  scalarProduct_ = ScalarProductChooser::construct(*matrices_->parallelInformation().coarsest());
	}
#if 0 && HAVE_SUPERLU
      // Use superlu if we are purely sequential or with only one processor on the coarsest level.
	if(is_same<ParallelInformation,SequentialInformation>::value // sequential mode 
	   || matrices_->parallelInformation().coarsest()->communicator().size()==1 //parallel mode and only one processor
	   || (matrices_->parallelInformation().coarsest().isRedistributed() 
	       && matrices_->parallelInformation().coarsest().getRedistributed().communicator().size()==1
	       && matrices_->parallelInformation().coarsest().getRedistributed().communicator().size()>0)){ // redistribute and 1 proc
	  if(verbosity_>0 && matrices_->parallelInformation().coarsest()->communicator().rank()==0)
	  std::cout<<"Using superlu"<<std::endl;
	  if(matrices_->parallelInformation().coarsest().isRedistributed())
	    {
	      if(matrices_->matrices().coarsest().getRedistributed().getmat().N()>0)
		// We are still participating on this level
		solver_  = new SuperLU<typename M::matrix_type>(matrices_->matrices().coarsest().getRedistributed().getmat());
	      else
		solver_ = 0;
	    }else
	      solver_  = new SuperLU<typename M::matrix_type>(matrices_->matrices().coarsest()->getmat());
	}else
#endif
	  {
	    if(matrices_->parallelInformation().coarsest().isRedistributed())
	      {
		if(matrices_->matrices().coarsest().getRedistributed().getmat().N()>0)
		  // We are still participating on this level
		  solver_ = new BiCGSTABSolver<X>(const_cast<M&>(matrices_->matrices().coarsest().getRedistributed()), 
						  *scalarProduct_, 
						  *coarseSmoother_, 1E-2, 10000, 0);
		else
		  solver_ = 0;
	      }else if (!solver_)
	      solver_ = new BiCGSTABSolver<X>(const_cast<M&>(*matrices_->matrices().coarsest()), 
					      *scalarProduct_, 
					      *coarseSmoother_, 1E-2, 1000, 0);
	  }
      }
      predone = true;
    }

    template<class M, class X, class S, class PI, class A>
    std::size_t AMG<M,X,S,PI,A>::levels()
    {
      return matrices_->levels();
    }
    template<class M, class X, class S, class PI, class A>
    std::size_t AMG<M,X,S,PI,A>::maxlevels()
    {
      return matrices_->maxlevels();
    }

    /** \copydoc Preconditioner::apply */
    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::apply(Domain& v, const Range& d)
    {
      ThreadContext& context = getThreadContext();

      if(additive){
	*(context.rhs_->finest())=d;
	additiveMgc();
	v=*context.lhs_->finest();
      }else{
        // Init all iterators for the current level
        initIteratorsWithFineLevel();

	
	*context.lhs = v;
	*context.rhs = d;
	*context.update=0;
	context.level=0;
		  
	mgc();
	
	if(postSteps_==0||matrices_->maxlevels()==1)
	  context.pinfo->copyOwnerToAll(*context.update, *context.update);
	
	v=*context.update;
      }
      
    }

    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::initIteratorsWithFineLevel()
    {
      ThreadContext& context = getThreadContext();

      context.smoother = smoothers_.finest();
      context.matrix = matrices_->matrices().finest();
      context.pinfo = matrices_->parallelInformation().finest();
      context.redist = 
        matrices_->redistributeInformation().begin();
      context.aggregates = matrices_->aggregatesMaps().begin();
      context.lhs = context.lhs_->finest();
      context.update = context.update_->finest();
      context.rhs = context.rhs_->finest();
    }
    
    template<class M, class X, class S, class PI, class A>
    bool AMG<M,X,S,PI,A>
    ::moveToCoarseLevel()
    {
      
      bool processNextLevel=true;
      ThreadContext& context = getThreadContext();
      
      if(context.redist->isSetup()){
        context.redist->redistribute(static_cast<const Range&>(*context.rhs), context.rhs.getRedistributed());
        processNextLevel =context.rhs.getRedistributed().size()>0;
        if(processNextLevel){		
          //restrict defect to coarse level right hand side.
          typename Hierarchy<Range,A>::Iterator fineRhs = context.rhs++;
          ++context.pinfo;
          Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
            ::restrict(*(*context.aggregates), *context.rhs, static_cast<const Range&>(fineRhs.getRedistributed()), *context.pinfo);
        }
      }else{		
        //restrict defect to coarse level right hand side.
        typename Hierarchy<Range,A>::Iterator fineRhs = context.rhs++;
	  ++context.pinfo;
	  Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
	    ::restrict(*(*context.aggregates), *context.rhs, static_cast<const Range&>(*fineRhs), *context.pinfo);
      }
      
      if(processNextLevel){
        // prepare coarse system
        ++context.lhs;
        ++context.update;
        ++context.matrix;
        ++context.level;
        ++context.redist;

        if(context.matrix != matrices_->matrices().coarsest() || matrices_->levels()<matrices_->maxlevels()){
          // next level is not the globally coarsest one
          ++context.smoother;
          ++context.aggregates;
        }
        // prepare the update on the next level
        *context.update=0;
      }
      return processNextLevel;
    }
    
    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>
    ::moveToFineLevel(bool processNextLevel)
    {
      ThreadContext& context = getThreadContext();
      if(processNextLevel){
        if(context.matrix != matrices_->matrices().coarsest() || matrices_->levels()<matrices_->maxlevels()){
          // previous level is not the globally coarsest one
	    --context.smoother;
	    --context.aggregates;
        }
        --context.redist;
        --context.level;
        //prolongate and add the correction (update is in coarse left hand side)
        --context.matrix;
	
        //typename Hierarchy<Domain,A>::Iterator coarseLhs = lhs--;
        --context.lhs;  
        --context.pinfo;
      }
      if(context.redist->isSetup()){
        // Need to redistribute during prolongate
        context.lhs.getRedistributed()=0;
        Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
          ::prolongate(*(*context.aggregates), *context.update, *context.lhs, context.lhs.getRedistributed(), matrices_->getProlongationDampingFactor(), 
                       *context.pinfo, *context.redist);
      }else{
        *context.lhs=0;
        Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
          ::prolongate(*(*context.aggregates), *context.update, *context.lhs, 
                       matrices_->getProlongationDampingFactor(), *context.pinfo);
      }
      
      
      if(processNextLevel){
        --context.update;
        --context.rhs;
      }
      
      *context.update += *context.lhs;
    }
    
    
    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>
    ::presmooth()
    {
      ThreadContext& context = getThreadContext();
      
      for(std::size_t i=0; i < preSteps_; ++i){
	    *context.lhs=0;
	    SmootherApplier<S>::preSmooth(*context.smoother, *context.lhs, *context.rhs);
	    // Accumulate update
	    *context.update += *context.lhs;
	    
	    // update defect
	    context.matrix->applyscaleadd(-1,static_cast<const Domain&>(*context.lhs), *context.rhs);
	    context.pinfo->project(*context.rhs);
          }
    }
    
     template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>
     ::postsmooth()
    { 
        ThreadContext& context = getThreadContext();
      
	for(std::size_t i=0; i < postSteps_; ++i){
	  // update defect
	  context.matrix->applyscaleadd(-1,static_cast<const Domain&>(*context.lhs), *context.rhs);
	  *context.lhs=0;
	  context.pinfo->project(*context.rhs);
	  SmootherApplier<S>::postSmooth(*context.smoother, *context.lhs, *context.rhs);
	  // Accumulate update
	  *context.update += *context.lhs;
        }
    }
    
    
    template<class M, class X, class S, class PI, class A>
    bool AMG<M,X,S,PI,A>::usesDirectCoarseLevelSolver() const
    {
      return IsDirectSolver< CoarseSolver>::value;
    }
    
    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::mgc(){
      ThreadContext& context = getThreadContext();
      if(context.matrix == matrices_->matrices().coarsest() && levels()==maxlevels()){
	// Solve directly
	InverseOperatorResult res;
	res.converged=true; // If we do not compute this flag will not get updated
	if(context.redist->isSetup()){
	    context.redist->redistribute(*context.rhs, context.rhs.getRedistributed());
	  if(context.rhs.getRedistributed().size()>0){
	    // We are still participating in the computation
	    context.pinfo.getRedistributed().copyOwnerToAll(context.rhs.getRedistributed(), context.rhs.getRedistributed());
	    solver_->apply(context.update.getRedistributed(), context.rhs.getRedistributed(), res);
	  }
	  context.redist->redistributeBackward(*context.update, context.update.getRedistributed());
	  context.pinfo->copyOwnerToAll(*context.update, *context.update);
	}else{
	  context.pinfo->copyOwnerToAll(*context.rhs, *context.rhs);
	  solver_->apply(*context.update, *context.rhs, res);
	}

	if (!res.converged)
	  coarsesolverconverged = false;
      }else{
	// presmoothing
        presmooth();

#ifndef DUNE_AMG_NO_COARSEGRIDCORRECTION
        bool processNextLevel = moveToCoarseLevel();
        
        if(processNextLevel){
          // next level
	  for(std::size_t i=0; i<gamma_; i++)
	    mgc();
        }
        
        moveToFineLevel(processNextLevel);
#else
        *lhs=0;
#endif	

	if(context.matrix == matrices_->matrices().finest()){
	  coarsesolverconverged = matrices_->parallelInformation().finest()->communicator().prod(coarsesolverconverged);
	  if(!coarsesolverconverged)
	    DUNE_THROW(MathError, "Coarse solver did not converge");
	}
	// postsmoothing
        postsmooth();
        
      }
    }

    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::additiveMgc(){
      ThreadContext& context = getThreadContext();
      
      // restrict residual to all levels
      typename ParallelInformationHierarchy::Iterator pinfo=matrices_->parallelInformation().finest();
      typename Hierarchy<Range,A>::Iterator rhs=context.rhs_->finest();      
      typename Hierarchy<Domain,A>::Iterator lhs = context.lhs_->finest();
      typename OperatorHierarchy::AggregatesMapList::const_iterator aggregates=matrices_->aggregatesMaps().begin();
      
      for(typename Hierarchy<Range,A>::Iterator fineRhs=rhs++; fineRhs != context.rhs_->coarsest(); fineRhs=rhs++, ++aggregates){
	++pinfo;
	Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
	  ::restrict(*(*context.aggregates), *context.rhs, static_cast<const Range&>(*fineRhs), *pinfo);
      }
      
      // pinfo is invalid, set to coarsest level
      //pinfo = matrices_->parallelInformation().coarsest
      // calculate correction for all levels
      lhs = context.lhs_->finest();
      typename Hierarchy<Smoother,A>::Iterator smoother = smoothers_.finest();
      
      for(rhs=context.rhs_->finest(); rhs != context.rhs_->coarsest(); ++lhs, ++rhs, ++smoother){
	// presmoothing
	*lhs=0;
	smoother->apply(*lhs, *rhs);
      }
      
      // Coarse level solve
#ifndef DUNE_AMG_NO_COARSEGRIDCORRECTION 
      InverseOperatorResult res;
      pinfo->copyOwnerToAll(*rhs, *rhs);
      solver_->apply(*lhs, *rhs, res);
      
      if(!res.converged)
	DUNE_THROW(MathError, "Coarse solver did not converge");
#else
      *lhs=0;
#endif
      // Prologate and add up corrections from all levels
      --pinfo;
      --aggregates;
      
      for(typename Hierarchy<Domain,A>::Iterator coarseLhs = lhs--; coarseLhs != context.lhs_->finest(); coarseLhs = lhs--, --aggregates, --pinfo){
	Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
	  ::prolongate(*(*aggregates), *coarseLhs, *lhs, 1, *pinfo);
      }
    }

    
    /** \copydoc Preconditioner::post */
    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::post(Domain& x)
    {
      ThreadContext& context = getThreadContext();

      if(buildHierarchy_){
//        if(solver_)
//          delete solver_;
//        if(coarseSmoother_)
//          ConstructionTraits<Smoother>::deconstruct(coarseSmoother_);
      }
      
      // Postprocess all smoothers
      typedef typename Hierarchy<Smoother,A>::Iterator Iterator;
      typedef typename Hierarchy<Range,A>::Iterator RIterator;
      typedef typename Hierarchy<Domain,A>::Iterator DIterator;
      Iterator coarsest = smoothers_.coarsest();
      Iterator smoother = smoothers_.finest();
      DIterator lhs = context.lhs_->finest();
      if(smoothers_.levels()>0){
	if(smoother != coarsest  || matrices_->levels()<matrices_->maxlevels())
	  smoother->post(*lhs);
	if(smoother!=coarsest)
	  for(++smoother, ++lhs; smoother != coarsest; ++smoother, ++lhs)
	    smoother->post(*lhs);
	smoother->post(*lhs);
      }

      delete &(*context.lhs_->finest());
      delete context.lhs_;
      delete &(*context.update_->finest());
      delete context.update_;
      delete &(*context.rhs_->finest());
      delete context.rhs_;
    }

    template<class M, class X, class S, class PI, class A>
    template<class A1>
    void AMG<M,X,S,PI,A>::getCoarsestAggregateNumbers(std::vector<std::size_t,A1>& cont)
    {
      matrices_->getCoarsestAggregatesOnFinest(cont);
    }
  } // end namespace Amg
}// end namespace Dune
  
#endif
