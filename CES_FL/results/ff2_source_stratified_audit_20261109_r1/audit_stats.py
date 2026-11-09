"""CPU statistics for the frozen-score diagnostic; no models or image I/O."""
import numpy as np


def connected_components(rows):
    """Union all recorded identities/masters, not just a convenient first-ID cluster."""
    parent=list(range(len(rows)));seen={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(rows):
        tokens=[('image',r['image'])]
        tokens += [(field,t) for field in ('identity_ids','source_ids') for t in r[field].split(';') if t]
        for t in tokens:
            if t in seen:parent[root(i)]=root(seen[t])
            else:seen[t]=i
    roots=[root(i) for i in range(len(rows))]
    return np.unique(roots,return_inverse=True)[1]


class AUC:
    def __init__(self,y,z,comp,ix):
        order=ix[np.argsort(z[ix],kind='stable')]
        self.comp=comp[order];self.y=y[order]
        self.start=np.flatnonzero(np.r_[True,np.diff(z[order])!=0]) if len(order) else np.array([],int)
    def __call__(self,weights):
        if not len(self.start):return np.nan
        w=weights[self.comp]
        real=np.add.reduceat(w*(self.y==0),self.start)
        fake=np.add.reduceat(w*(self.y==1),self.start)
        den=real.sum()*fake.sum()
        return float((fake*(np.cumsum(real)-.5*real)).sum()/den) if den else np.nan


def build_cohorts(rows,source,comp,groups):
    y=np.array([r['label'] for r in rows]);n=len(rows);coh=[]
    def add(name,mask,kind,axis='',group='',src='ALL',auc_boot=False):
        ix=np.flatnonzero(mask);ri=ix[y[ix]==0];fi=ix[y[ix]==1]
        coh.append(dict(name=name,ix=ix,real=ri,fake=fi,kind=kind,axis=axis,group=group,source=src,
                        n=len(ix),n_real=len(ri),n_fake=len(fi),n_components=len(np.unique(comp[ix])),
                        n_real_components=len(np.unique(comp[ri])),n_fake_components=len(np.unique(comp[fi])),auc_boot=auc_boot))
    add('overall',np.ones(n,bool),'overall',auc_boot=True)
    for ax,names in groups.items():
        for g in names:add(ax+'='+g,np.array([r[ax]==g for r in rows]),'group',ax,g,auc_boot=ax=='gender_race')
    for ax in ('age_bin','skin_bin'):
        for g in sorted({r[ax] for r in rows}):add(ax+'='+g,np.array([r[ax]==g for r in rows]),'proxy_group',ax,g)
    for s in sorted(set(source)):
        sm=source==s;add('source='+s,sm,'source',src=s)
        for ax,names in groups.items():
            for g in names:add('source='+s+'/'+ax+'='+g,sm&np.array([r[ax]==g for r in rows]),'source_group',ax,g,s)
    families=sorted({r['family'] for r in rows if r['label']==1})
    for fam in families:
        fm=np.array([r['family']==fam for r in rows])
        add('family='+fam,(y==0)|((y==1)&fm),'family',group=fam,auc_boot=True)
        for s in sorted(set(source)):
            add('source='+s+'/family='+fam,(source==s)&((y==0)|((y==1)&fm)),'source_family',group=fam,src=s)
    return coh


def run_statistics(rows,source,components,z,thresholds,names,freeze,progress=None):
    """One paired component-resampling sequence shared by every model/seed."""
    group_names=freeze['groups'];sources=freeze['sources'];w=np.array([freeze['source_weights'][s] for s in sources])
    y=np.array([r['label'] for r in rows]);source=np.array(source);comp=np.asarray(components)
    nc=int(comp.max())+1;nm=len(names);coh=build_cohorts(rows,source,comp,group_names);nk=len(coh)
    lookup={c['name']:k for k,c in enumerate(coh)}
    dr=np.zeros((nc,nk));df=np.zeros_like(dr);fp=np.zeros((nm,nc,nk));fn=np.zeros_like(fp)
    auc_plans={};keys=[];supports={}
    def key(metric,cohort):return metric+'/'+cohort
    for k,c in enumerate(coh):
        ri,fi=c['real'],c['fake'];dr[:,k]=np.bincount(comp[ri],minlength=nc);df[:,k]=np.bincount(comp[fi],minlength=nc)
        for m in range(nm):
            fp[m,:,k]=np.bincount(comp[ri],weights=(z[m,ri]>=thresholds[m]),minlength=nc)
            fn[m,:,k]=np.bincount(comp[fi],weights=(z[m,fi]<thresholds[m]),minlength=nc)
            auc_plans[m,k]=AUC(y,z[m],comp,c['ix'])
        for metric in ('FPR','FNR','AUC'):
            q=key(metric,c['name']);keys.append(q);supports[q]={k:v for k,v in c.items() if k not in ('ix','real','fake','auc_boot')}
    contrasts=[];standardized_groups=[];coverage={}
    for ax,gs in group_names.items():
        gi=np.array([lookup[ax+'='+g] for g in gs]);si=np.array([[lookup['source='+s+'/'+ax+'='+g] for s in sources] for g in gs])
        missing=[{'source':s,'group':g,'n_real':coh[si[i,j]]['n_real']} for i,g in enumerate(gs) for j,s in enumerate(sources) if not coh[si[i,j]]['n_real']]
        covered=sum(coh[k]['n_real'] for k in gi)==int((y==0).sum())
        coverage[ax]={'missing_real_cells':missing,'complete_real_group_coverage':covered,'source_standardization_identified':not missing and covered}
        contrasts.append((ax,gi,si))
        for label in ['D_native','D_standardized','native_minus_standardized','native_FPR_gap']:
            keys.append(label+'/'+ax)
        for s in sources:
            keys += ['within_source_D/'+ax+'/'+s,'within_source_FPR_gap/'+ax+'/'+s]
        for i,g in enumerate(gs):keys.append('standardized_FPR/'+ax+'/'+g)
    def compute(weights,boot=False):
        nr=weights@dr;nf=weights@df
        fp_count=(weights@fp.transpose(1,0,2).reshape(nc,-1)).reshape(nm,nk)
        fn_count=(weights@fn.transpose(1,0,2).reshape(nc,-1)).reshape(nm,nk)
        with np.errstate(divide='ignore',invalid='ignore'):
            pr=fp_count/nr;fr=fn_count/nf
        out={}
        for k,c in enumerate(coh):
            out[key('FPR',c['name'])]=pr[:,k];out[key('FNR',c['name'])]=fr[:,k]
            out[key('AUC',c['name'])]=np.array([auc_plans[m,k](weights) for m in range(nm)]) if (not boot or c['auc_boot']) else np.full(nm,np.nan)
        for ax,gi,si in contrasts:
            q=pr[:,gi];qgs=pr[:,si];pi=nr[gi]/nr[lookup['overall']]
            ok=np.isfinite(q).all(axis=1)
            native=np.abs(q-pr[:,lookup['overall'],None]).sum(1)
            native[~ok]=np.nan;out['D_native/'+ax]=native
            out['native_FPR_gap/'+ax]=np.where(ok,q.max(1)-q.min(1),np.nan)
            qstd=(qgs*w[None,None,:]).sum(2)
            st=np.abs(qstd-(qstd*pi[None,:]).sum(1)[:,None]).sum(1)
            valid=np.isfinite(qstd).all(1)&coverage[ax]['complete_real_group_coverage']
            st[~valid]=np.nan;out['D_standardized/'+ax]=st;out['native_minus_standardized/'+ax]=native-st
            for i,g in enumerate(group_names[ax]):out['standardized_FPR/'+ax+'/'+g]=qstd[:,i]
            for j,s in enumerate(sources):
                a=qgs[:,:,j];ok=np.isfinite(a).all(1);pool=pr[:,lookup['source='+s]]
                out['within_source_D/'+ax+'/'+s]=np.where(ok,np.abs(a-pool[:,None]).sum(1),np.nan)
                out['within_source_FPR_gap/'+ax+'/'+s]=np.where(ok,a.max(1)-a.min(1),np.nan)
        return np.stack([out[k] for k in keys],axis=1)
    point=compute(np.ones(nc));reps=freeze['bootstrap']['replicates'];draws=np.full((reps,nm,len(keys)),np.nan)
    rng=np.random.default_rng(freeze['bootstrap']['seed'])
    for b in range(reps):
        counts=rng.multinomial(nc,np.full(nc,1/nc))
        draws[b]=compute(counts,True)
        if progress and (b+1)%100==0:progress(b+1,reps)
    ci=[];minimum=int(np.ceil(reps*freeze['bootstrap']['required_valid_fraction']))
    def record(model,metric,p,v,kind='model'):
        good=v[np.isfinite(v)];d={'model':model,'metric':metric,'estimate':float(p) if np.isfinite(p) else None,
            'valid_replicates':len(good),'invalid_replicates':reps-len(good),'required_valid_replicates':minimum,
            'ci95_low':None,'ci95_high':None,'interval_role':'exploratory unadjusted; conditional on selected models','kind':kind}
        if not np.isfinite(p):d['status']='BLOCKED_UNDEFINED_ESTIMAND'
        elif metric.startswith('AUC/') and metric[4:] in lookup and not coh[lookup[metric[4:]]]['auc_boot']:
            d['status']='POINT_ONLY';d['reason']='AUC interval not prespecified for this supplementary source/proxy stratum.'
        elif len(good)<minimum:d['status']='INCONCLUSIVE_BOOTSTRAP_SUPPORT'
        else:
            d['ci95_low'],d['ci95_high']=map(float,np.quantile(good,[.025,.975]));d['status']='DESCRIPTIVE_INTERVAL'
            if kind=='paired' or metric.startswith('native_minus_standardized/'):
                d['status']='INCONCLUSIVE_INTERVAL_SPANS_ZERO' if d['ci95_low']<=0<=d['ci95_high'] else 'DIRECTIONAL_ASSOCIATION_ONLY'
        ci.append(d)
    for m,name in enumerate(names):
        for k,metric in enumerate(keys):record(name,metric,point[m,k],draws[:,m,k])
    for seed in (17,7,27):
        s=names.index('RF-S_s'+str(seed));j=names.index('RF-SJg_s'+str(seed))
        for k,metric in enumerate(keys):
            if metric.startswith(('D_native/','D_standardized/','native_minus_standardized/','within_source_D/')) or metric in ('AUC/overall','FPR/overall','FNR/overall'):
                record('SJg-minus-S_s'+str(seed),metric,point[j,k]-point[s,k],draws[:,j,k]-draws[:,s,k],'paired')
    # The one declared primary interval is the seed17 RF-S joint source-standardization contrast.
    primary=next(x for x in ci if x['model']=='RF-S_s17' and x['metric']=='native_minus_standardized/gender_race')
    primary['interval_role']='prespecified primary 95% interval; no multiplicity adjustment'
    return {'coverage':coverage,'intervals':ci,'cohorts':[{k:v for k,v in c.items() if k not in ('ix','real','fake')} for c in coh],
            'components':{'n':nc,'largest_component_rows':int(np.bincount(comp).max()),
                          'row_size_histogram':{str(k):int(v) for k,v in zip(*np.unique(np.bincount(comp),return_counts=True))}},
            'keys':keys,'point':point,'draws':draws,'primary':primary}
