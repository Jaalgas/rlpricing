"""Fit a conditional logit to Tjiong mobile-data stated choices.
Usage: python esim_logit.py tjiong_2026_mobile_data.csv
Dependencies: pip install pandas numpy scipy scikit-learn
Prices in original PLN units; no conversion to present-day eSIM willingness to pay implied.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import GroupShuffleSplit

FEATURES=['mobile_constant','price_pln','log_data_gb','multi_device','free_mobile','wifi_price_pln','mobile_SP2','mobile_SP3','free_mobile_SP2','free_mobile_SP3']

def numeric(s):
    return pd.to_numeric(s.str.extract(r'(\d+(?:[.,]\d+)?)')[0].str.replace(',','.',regex=False),errors='coerce')

def prepare(df):
    df=df.copy()
    mobile=df.attr_data_limit.ne('-')
    price=numeric(df.attr_data_fee); gb=numeric(df.attr_data_limit); devices=numeric(df.attr_devices); wifi=numeric(df.attr_wifi_fee)
    if (price[mobile].isna().any() or gb[mobile].isna().any() or devices[mobile].isna().any() or wifi[~mobile].isna().any()): raise ValueError('Unrecognized attribute levels')
    free=mobile & price.eq(0)
    df['mobile_constant']=mobile.astype(float)
    df['price_pln']=price.where(mobile,0)
    df['log_data_gb']=np.log1p(gb.where(mobile,0))
    df['multi_device']=(mobile & devices.gt(1)).astype(float)
    df['free_mobile']=free.astype(float)
    df['wifi_price_pln']=wifi.where(~mobile,0)
    for t in ['SP2','SP3']:
        df['mobile_'+t]=(mobile & df.trial_treatment.eq(t)).astype(float)
        df['free_mobile_'+t]=(free & df.trial_treatment.eq(t)).astype(float)
    return df

def arrays(df, features):
    groups=list(df.groupby(['id','task','trial_treatment','trial_block'],sort=False))
    m=max(len(g) for _,g in groups); X=np.zeros((len(groups),m,len(features))); y=np.zeros((len(groups),m)); mask=np.zeros_like(y,dtype=bool)
    for i,(_,g) in enumerate(groups):
        n=len(g); assert g.choice.sum()==1 and g.profile.nunique()==n
        X[i,:n]=g[features].to_numpy(); y[i,:n]=g.choice; mask[i,:n]=True
    return X,y,mask

def objective(b,X,y,mask,gradient=True):
    u=np.where(mask,X@b,-np.inf); lse=logsumexp(u,axis=1); chosen=np.sum(np.where(mask,X@b,0)*y,axis=1); loss=np.sum(lse-chosen)
    p=np.exp(u-lse[:,None]); grad=np.einsum('ij,ijk->k',p-y,X)
    return (loss,grad) if gradient else (loss,p)

def fit(df, features):
    X,y,m=arrays(df,features)
    # Remove unidentifiable features (constant within choice sets), or collinear contrasts.
    centered=(X-(X*m[:,:,None]).sum(1)[:,None,:]/m.sum(1)[:,None,None])*m[:,:,None]
    if np.linalg.matrix_rank(centered.reshape(-1,len(features)))<len(features): raise ValueError('Choice-set feature contrasts are rank deficient')
    result=minimize(objective,np.zeros(len(features)),args=(X,y,m),jac=True,method='BFGS',options={'gtol':1e-5,'maxiter':1000})
    if not result.success and np.max(np.abs(result.jac))>1e-3: raise RuntimeError(result.message)
    return result.x

def evaluate(df,features,b):
    X,y,m=arrays(df,features); loss,p=objective(b,X,y,m,False); n=len(y); null=np.log(m.sum(1)).sum()
    return {'choice_sets':n,'log_loss':float(loss/n),'accuracy':float(np.mean(p.argmax(1)==y.argmax(1))),'mcfadden_r2_vs_uniform':float(1-loss/null)}

def esim_probabilities(offers, coefficients, *, outside_utility, validity_weight=0., quality_weight=0., brand_weight=0.):
    """Scenario transfer, NOT calibrated eSIM demand. price_pln must be in original model units.
    Optional weights are user assumptions, not estimated. Quality and brand scores should be 0..1.
    Outside utility must be chosen/calibrated separately: survey outside alternative was Wi-Fi.
    """
    v=[]
    for o in offers:
        p=float(o['price_pln']); g=float(o['data_gb']); d=float(o.get('devices',1))
        if p<0 or g<=0: raise ValueError('Invalid offer')
        u=coefficients.get('mobile_constant',0)+coefficients['price_pln']*p+coefficients['log_data_gb']*np.log1p(g)+coefficients.get('multi_device',0)*(d>1)+coefficients.get('free_mobile',0)*(p==0)
        u+=validity_weight*np.log1p(o.get('validity_days',0))+quality_weight*o.get('quality',0)+brand_weight*o.get('brand',0)
        v.append(u)
    v=np.array(v+[outside_utility]); return np.exp(v-logsumexp(v))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('csv',type=Path); ap.add_argument('--output',type=Path,default=Path('esim_logit_results.json')); args=ap.parse_args()
    df=prepare(pd.read_csv(args.csv)); tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,groups=df.id)); train,test=df.iloc[tr],df.iloc[te]
    # SP3 has no Wi-Fi option; its mobile constant cancels within those sets.
    features=[f for f in FEATURES if f not in ('mobile_SP3','free_mobile_SP2','free_mobile_SP3')]
    b=fit(train,features); metrics=evaluate(test,features,b); full=fit(df,features)
    out={'features':features,'coefficients':dict(zip(features,map(float,full))),'held_out_metrics':metrics,'training_coefficients':dict(zip(features,map(float,b))),'note':'Stated choices; PLN and original survey units. Validity, coverage, brand and true no-purchase utility not identified.'}
    args.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
