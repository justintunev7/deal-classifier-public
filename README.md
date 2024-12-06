# T2-deal-classifier

This classifier was built to classify deals into 3 categories: "Good", "Bad" and "Neutral". The classifier was built using a dataset of 1000 deals, which were manually classified by a team. The classifier was built using a Random Forest Classifier, which was trained on the dataset of 1000 deals. The classifier was then tested on a separate dataset of 100 deals, which were also manually classified by the same team. The classifier achieved an accuracy of >85% on the test dataset.

This repo is the experimentation while building the classifier, and the final classifier is not included in this repo. Some code is commented out to show some experimentation with different ML models and weights that were tried while building the classifier.

Data is omitted for privacy reasons, but these are the columns that were used in the dataset:
```
apr_rate,balloon_amount,bank_name,deal_book_date,lease_annual_miles,lease_net_cap_cost,lease_payment,lease_term,license_fee,residual_amount,retail_payment,salesman_1_name,sale_type,salesman_commission,total_tax,vehicle_type,deal_type
```

The classifier was built using Python and the following libraries:
- pandas
- numpy
- scikit-learn

