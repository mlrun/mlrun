class LearningCurves(MLPlotPlan):
    """
    Plot cross-validated training and test scores for different training set sizes.
    """

    _ARTIFACT_NAME = "learning_curves"

    def __init__(
            self,
            model=None,
            X_train=None,
            y_train=None,
            cv: int = 3,
            groups=None,
            train_sizes: np.array = np.array([0.1, 0.33, 0.55, 0.78, 1.0]),
            scoring=None,
            exploit_incremental_learning: bool = False,
            n_jobs=None,
            pre_dispatch: str = "all",
            verbose: int = 0,
            shuffle: bool = False,
            random_state=None,
            return_times: bool = True,
            fit_params=None,
    ):
        """

        :param model: a fitted model.
        :param X_train: training dataset.
        :param y_train: target dataset.
        :param cv: Determines the cross-validation splitting strategy.
        :param groups: Group labels for the samples used while splitting the dataset into train/test set.
        :param train_sizes: Relative or absolute numbers of training examples that will be used to generate the learning curve.
        :param scoring: A str (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).
        :param exploit_incremental_learning: If the estimator supports incremental learning, this will be used to speed up fitting for different training set sizes.
        :param n_jobs: Number of jobs to run in parallel.
        :param pre_dispatch: Number of predispatched jobs for parallel execution (default is all).
        :param verbose: Controls the verbosity: the higher, the more messages.
        :param shuffle: Whether to shuffle training data before taking prefixes of it based on``train_sizes``.
        :param random_state: Used when shuffle is True. Pass an int for reproducible output across multiple function calls.
        :param return_times: Value to assign to the score if an error occurs in estimator fitting.
        :param fit_params: Parameters to pass to the fit method of the estimator.
        """

        # learning_curve() params
        self._groups = groups
        self._cv = cv
        self._train_sizes = train_sizes
        self._scoring = scoring
        self._exploit_incremental_learning = exploit_incremental_learning
        self._n_jobs = n_jobs
        self._pre_dispatch = pre_dispatch
        self._verbose = verbose
        self._shuffle = shuffle
        self._random_state = random_state
        self._return_times = return_times
        self._fit_params = fit_params

        super(LearningCurves, self).__init__(
            model=model, X_train=X_train, y_train=y_train
        )

    def is_ready(self, stage: MLPlanStages, is_probabilities: bool) -> bool:
        """
        Check whether or not the plan is fit for production by the given stage and prediction probabilities. The
        confusion matrix is ready only post prediction.
        :param stage:            The stage to check if the plan is ready.
        :param is_probabilities: True if the 'y_pred' that will be sent to 'produce' is a prediction of probabilities
                                 (from 'predict_proba') and False if not.
        :return: True if the plan is producible and False otherwise.
        """
        return stage == MLPlanStages.POST_FIT and not is_probabilities

    def produce(self, model, X_train, y_train, **kwargs) -> Dict[str, PlotlyArtifact]:
        validate_numerical(X_train)
        validate_numerical(y_train)

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            model,
            X_train,
            y_train.values.ravel(),
            groups=self._groups,
            cv=self._cv,
            train_sizes=self._train_sizes,
            scoring=self._scoring,
            _exploit_incremental_learning=self._exploit_incremental_learning,
            n_jobs=self._n_jobs,
            pre_dispatch=self._pre_dispatch,
            verbose=self._verbose,
            shuffle=self._shuffle,
            random_state=self._random_state,
            return_times=self._return_times,
            fit_params=self._fit_params,
        )

        fig = go.Figure(
            data=[go.Scatter(x=train_sizes.tolist(), y=np.mean(train_scores, axis=1))],
            layout={'title': {'text': 'Learning Curves'}},
        )

        # add custom xaxis title
        fig.add_annotation(
            {'font': {'color': 'black', 'size': 14},
             'x': 0.5,
             'y': -0.15,
             'showarrow': False,
             'text': 'Train Size',
             'xref': 'paper',
             'yref': 'paper'}
        )

        # add custom yaxis title
        fig.add_annotation(
            {'font': {'color': 'black', 'size': 14},
             'x': -0.1,
             'y': 0.5,
             'showarrow': False,
             'text': 'Score',
             'textangle': -90,
             'xref': 'paper',
             'yref': 'paper'}
        )

        # adjust margins to make room for yaxis title
        fig.update_layout(margin={'t': 100, 'l': 100}, width=800, height=500)

        # Creating an html rendering of the plot
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            figure=fig, key=self._ARTIFACT_NAME
        )
        return self._artifacts