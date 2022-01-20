# use a package which is not in mlrun image (to test build)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def handler(context, text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    print("score:", str(score))
    context.log_result("score", str(score))
