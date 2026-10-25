# Frozen Effusion finalist comparison

Status: complete.

Frozen selected policies on repeatedly used development-selection cohort; separate conditional training-seed and realized-head patient intervals; no joint uncertainty, untouched-cohort confirmation, clinical claim, family-wide discovery or whole-study DP claim.

AUROC gains are left minus right. Native anchors are excluded from all main-arm means.

## Primary

Projected128 minus mean512, private clean: 0.01917564 AUROC; approximate paired-seed 95% t interval [0.01101125, 0.02734003].
Technical margin: +0.01; conclusion: practically_positive_seed_mean_gain. No clinical-benefit claim.

Paired sample SD: 0.01284983; standard error: 0.00370943; degrees of freedom: 11.

The seed interval assumes independent seed effects and an adequate t approximation. Twelve seeds do not certify those assumptions.

| Seed | Paired AUROC gain |
| --- | ---: |
| 13001 | 0.01493701 |
| 13002 | 0.01679650 |
| 13003 | -0.00525346 |
| 13004 | 0.02045449 |
| 13005 | 0.03108538 |
| 13006 | 0.01696561 |
| 13007 | 0.02089885 |
| 13008 | 0.02518702 |
| 13009 | 0.01475443 |
| 13010 | 0.00041941 |
| 13011 | 0.03874208 |
| 13012 | 0.03512038 |

Separate conditional patient interval: [0.01313594, 0.02589349]; 2000 valid draws and 0 skipped single-class draws.
patient-cluster percentile interval conditional on realized heads; image-level AUROC; not training-seed or joint uncertainty

## Main arm AUROC

| Arm | Completed seeds | Mean | Range | Sample SD |
| --- | ---: | ---: | --- | ---: |
| mean512:nonprivate:clean | 12 | 0.807298354463237 | 0.8063715464156154 to 0.8082928926899546 | 0.0005545445493434833 |
| mean512:private:clean | 12 | 0.7403606558767306 | 0.7173015698020138 to 0.7623983330207166 | 0.012307415941943537 |
| mean512:nonprivate:owned_pgd | 12 | 0.5285448512426186 | 0.5110191815359197 to 0.5698158570556285 | 0.01673047675075088 |
| projected128:nonprivate:clean | 12 | 0.7923752976926883 | 0.7919327027453079 to 0.7927497675328147 | 0.00024308087505074498 |
| projected128:nonprivate:owned_pgd | 12 | 0.7618389432670319 | 0.7353651056777866 to 0.7819999062149552 | 0.015556236143910619 |
| trim16:nonprivate:clean | 12 | 0.8070020388270085 | 0.8058713179392831 to 0.8077538781113687 | 0.0005611108474713285 |
| trim16:nonprivate:owned_pgd | 12 | 0.5994418864878366 | 0.57486914990841 to 0.6331876090001716 | 0.015386209676630663 |
| center_quarter:nonprivate:clean | 12 | 0.8051066279005187 | 0.8039554590877708 to 0.8059653524122711 | 0.0006155667177013723 |
| center_quarter:nonprivate:owned_pgd | 12 | 0.7227250194637225 | 0.6757535478682858 to 0.786538254321096 | 0.03342155651271646 |
| mean512:private:owned_pgd | 12 | 0.5513040725241081 | 0.45358812601517323 to 0.7072264744206279 | 0.07561387701011209 |
| projected128:private:clean | 12 | 0.7595362990525715 | 0.7465746010144748 to 0.767089705395234 | 0.006478699547966811 |
| projected128:private:owned_pgd | 12 | 0.6833307122575057 | 0.6067162812832987 to 0.7244138933564269 | 0.033973129545813316 |
| trim16:private:clean | 12 | 0.678365477991111 | 0.6334359148830879 to 0.7234376309498628 | 0.023743024971000674 |
| trim16:private:owned_pgd | 12 | 0.6051138394965405 | 0.5138480103602543 to 0.6887079813068446 | 0.062228409390279904 |
| center_quarter:private:clean | 12 | 0.6664966085232646 | 0.601930749721638 to 0.7353144093737154 | 0.04141740138224144 |
| center_quarter:private:owned_pgd | 12 | 0.5083004441320493 | 0.37855759598999095 to 0.6068245331699744 | 0.06716852212387106 |

All five metrics and every per-seed value are in summary.json and arms.csv. Secondary paired AUROC intervals in contrasts.csv are descriptive and unadjusted; the interaction must be interpreted alongside absolute clean/attacked components.

## Failure accounting

All 196 declared outcomes completed.
