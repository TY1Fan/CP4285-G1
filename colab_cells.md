## Colab cells to add

Place these **three code cells immediately after notebook cell 31**, after the existing `calendar_data` and monthly drift plots. Run the notebook through cell 31 first in Colab so `calendar_data`, `TRAIN_START`, `TRAIN_END`, and `TEST_YEAR` exist. These are reference copies; the repository notebook has not been edited.

### Cell A — training-only lag-12 annual-repeat plot

This compares each month in 2015–2017 with the same month one year earlier. The rate is divided by each year's mean first, so an overall change in activity level cannot by itself create a strong lag-12 correlation. The right panel shows whether lag 12 stands out from other lags. Only training years enter either panel.

```python
train_monthly = calendar_data.loc[
    calendar_data.year.between(TRAIN_START, TRAIN_END)
].sort_values('date').copy()
assert (train_monthly.groupby('year').size() == 12).all()
assert len(train_monthly) >= 36

train_monthly['relative_rate'] = (
    train_monthly.events_per_active_user_day
    / train_monthly.groupby('year').events_per_active_user_day.transform('mean')
)
current = train_monthly.relative_rate
previous_year = current.shift(12)
paired = previous_year.notna()
lag12_r = current[paired].corr(previous_year[paired])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for year in range(TRAIN_START + 1, TRAIN_END + 1):
    year_pairs = train_monthly.year.eq(year) & paired
    axes[0].scatter(previous_year[year_pairs], current[year_pairs], label=str(year))
lo = min(previous_year[paired].min(), current[paired].min())
hi = max(previous_year[paired].max(), current[paired].max())
axes[0].plot([lo, hi], [lo, hi], '--', color='gray', linewidth=1)
axes[0].set(xlabel='Same month, previous year / that year mean',
            ylabel='Current month / current year mean',
            title=f'Training-only lag 12: r = {lag12_r:.2f} ({paired.sum()} pairs)')
axes[0].legend(title='Current year')

lags = np.arange(1, 25)
correlations = [current.autocorr(lag=int(lag)) for lag in lags]
axes[1].bar(lags, correlations,
            color=['#E3A345' if lag == 12 else '#246B8E' for lag in lags])
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set(xlabel='Lag in months', ylabel='Autocorrelation',
            title='Training-only within-year-normalized lag correlations')
axes[1].set_xticks([1, 3, 6, 9, 12, 15, 18, 21, 24])
plt.tight_layout()
plt.show()
```

**Observed Colab result:** The user-provided plot gives **lag-12 r = −0.19** across 36 pairs; lag 12 does not stand out as a positive correlation. This does **not support an annual-seasonality claim** for this normalized aggregate activity rate. It also does not prove that no seasonal pattern exists: four training years and correlated month pairs provide limited evidence. The year split is justified as chronological evaluation with complete 12-month validation and test windows. Chronological month- or day-level splits could also be valid.

### Cell B — decompose the monthly activity rate

This separates broad trend, repeating month-of-year structure, and irregular variation in *log(events per active user per day)*. It is an aggregate behavior diagnostic, not recommender residuals. Six years supply only six annual cycles, so treat apparent seasonality cautiously.

```python
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf

activity = (calendar_data.set_index('date')['events_per_active_user_day']
            .sort_index().asfreq('MS'))
assert len(activity) >= 36 and activity.notna().all()
log_activity = np.log1p(activity)

stl = STL(log_activity, period=12, robust=True).fit()
fig = stl.plot()
fig.set_size_inches(12, 8)
fig.suptitle('Monthly logged activity: STL decomposition', y=1.01)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].plot(stl.resid.index, stl.resid)
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].set_title('STL remainder over time')
axes[0].set_ylabel('Log-rate remainder')
plot_acf(stl.resid.dropna(), lags=12, ax=axes[1], zero=False)
axes[1].set_title('Remainder autocorrelation')
plt.tight_layout()
plt.show()

seasonal_strength = max(0, 1 - np.var(stl.resid) / np.var(stl.resid + stl.seasonal))
print(f'Descriptive seasonal strength: {seasonal_strength:.3f}')
```

Interpretation: a visible trend or repeatable seasonal component makes a frozen 2017 profile less representative of 2019 activity; large remainder swings indicate other changes. None of these alone explains ranking errors.

### Cell C — out-of-time seasonal-forecast residuals

This tests how well a **2017 same-month activity profile** transfers to 2018–2019, and compares it with a rolling one-year-lag reference. The rolling line uses the previous year's observed activity, so it is a diagnostic of freshness rather than a 2017-frozen forecast. This uses aggregate activity only; it does not score recommended songs.

```python
rate = (calendar_data.set_index('ym')['events_per_active_user_day']
        .sort_index().astype(float))
holdout = rate.loc[f'{TRAIN_END + 1}-01':f'{TEST_YEAR}-12']
assert len(holdout) > 0 and all(f'{TRAIN_END}-{m:02d}' in rate.index for m in range(1, 13))

frozen = pd.Series(
    [rate.loc[f'{TRAIN_END}-{int(ym[-2:]):02d}'] for ym in holdout.index],
    index=holdout.index, name='2017-profile forecast'
)
rolling = rate.shift(12).loc[holdout.index].rename('previous-year forecast')
assert rolling.notna().all()

residuals = pd.DataFrame({
    'actual': holdout,
    'frozen_prediction': frozen,
    'rolling_prediction': rolling,
})
residuals['frozen_error'] = residuals.actual - residuals.frozen_prediction
residuals['rolling_error'] = residuals.actual - residuals.rolling_prediction
residuals['year'] = residuals.index.str[:4]

summary = pd.DataFrame([
    {
        'year': year,
        'frozen_MAE': g.frozen_error.abs().mean(),
        'rolling_MAE': g.rolling_error.abs().mean(),
        'frozen_bias': g.frozen_error.mean(),
        'rolling_bias': g.rolling_error.mean(),
    }
    for year, g in residuals.groupby('year')
]).set_index('year')
display(summary.round(3))

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
residuals[['actual', 'frozen_prediction', 'rolling_prediction']].plot(ax=axes[0])
axes[0].set_ylabel('Events / active user / day')
axes[0].set_title('Observed monthly activity and time-respecting references')
residuals[['frozen_error', 'rolling_error']].plot(ax=axes[1])
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_ylabel('Actual minus prediction')
axes[1].set_xlabel('Month')
plt.tight_layout()
plt.show()
```

If 2019 frozen errors grow while rolling errors stay smaller, that would support the need to update *aggregate activity expectations*. To link temporal change to **recommendation** performance, the next analysis must save month-level model predictions and labels, then plot Precision@20, Recall@20, NDCG@20, and hit rate by month and training-history age on the **same eligible query set**. The current notebook does not save those predictions, so a recommendation-error residual plot cannot be honestly derived from its aggregate counts.

