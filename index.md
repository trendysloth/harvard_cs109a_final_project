---
title: Yelp Restaurant Recommendations
---

<!-- This is the home page

## Lets have fun

>here is a quote

Here is *emph* and **bold**.

Here is some inline math $\alpha = \frac{\beta}{\gamma}$ and, of-course, E rules:

$$ G_{\mu\nu} + \Lambda g_{\mu\nu}  = 8 \pi T_{\mu\nu} . $$ -->

Final Team Report:

Problem Statement and Motivation: This should be brief and self-contained. 
>Our goal for this project is to build a recommender system for restaurants using publicly available Yelp data. Our motivation is to understand the function of these systems and implement our own approaches based on what we learn from the EDA and what we have learned in the course thus far.

Introduction and Description of Data: Description of relevant knowledge. Why is this problem important? Why is it challenging? Introduce the motivations for the project question and how that question was defined through preliminary EDA. 
>Recommender systems are present in almost every popular website and app today. Most recently they have received a great deal of media attention due to the way that news articles and other web content are served to users based on their previous online behavior.



Literature Review/Related Work: This could include noting any key papers, texts, or websites that you have used to develop your modeling approach, as well as what others have done on this problem in the past. You must properly credit sources. 

>https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html
>http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/
>https://beckernick.github.io/matrix-factorization-recommender/


 Modeling Approach and Project Trajectory: Include 1) a baseline model for comparison and 2) a description of your implementations beyond the baseline model. Briefly summarize any changes in your project goals or implementation plans you have made along the way. These changes are a natural part of any project, even those that seem the most straightforward at the beginning. The story you tell about how you arrived at your results can powerfully illustrate your process. 

- Baseline model 1 (arithmetic)
- Baseline model 2 (ridge regression)
- Collaborative filtering with user-user + restaurant-restaurant similarity matrices
- Ensemble methods : averaging, boosting, linear regression

Results, Conclusions, and Future Work: Show and interpret your results. Summarize your results, the strengths and short-comings of your results, and speculate on how you might address these short-comings if given more time.

>If given more time, we would try out more ensemble methods such as 
