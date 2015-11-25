function   location = affgeo2loc(geo_param, p_sz)
w = geo_param(3)*p_sz;
h = w*geo_param(5);
tlx = geo_param(1) - (w-1)/2;
tly = geo_param(2) - (h-1)/2;
location = round([tlx, tly, w, h]);
end