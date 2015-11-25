function geo_params = drawparticals(geo_param, pf_param)
geo_param = repmat(geo_param, [1,pf_param.p_num]);
geo_params = geo_param + randn(6,pf_param.p_num).*repmat(pf_param.affsig(:),[1,pf_param.p_num]);
end