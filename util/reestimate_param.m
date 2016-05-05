function pf_param = reestimate_param(pf_param)
if pf_param.minconf > 0.49 %% low confidence
    scale = false;
elseif pf_param.minconf > 0.45 &&  pf_param.minconf <0.49%% high confidence
    scale = true;
elseif pf_param.minconf > 0.4 && pf_param.ratio > 0.6 %% median confidence and resolution
    scale = true;
elseif pf_param.minconf > 0.35 && pf_param.ratio < 0.3 %% high confidence with low resolution
    scale = true;
else 
    scale = false;
end

if scale
    pf_param.affsig = pf_param.affsig_o;
end
end