oggm_result_dir = '/home/www/lschuster/provide/MESMER-M_projections/runs/output/oggm_v16/2023.3/2100/'
provide_regions = [f'P{region + 1:02d}' for region in range(12)]
raw_oggm_output_file = 'run_hydro_w5e5_gcm_merged_*_*_q*_bc_2000_2019_Batch_*.nc'  # scenario, gcm, quantile
gcms_mesmer = [# 'ACCESS-CM2',  precipitation data not everywhere available ,
               # 'MCM-UA-1-0', no precipitation data
               'ACCESS-ESM1-5', 'CESM2-WACCM', 'CESM2', 'CMCC-CM2-SR5', 'CNRM-CM6-1-HR',
               'CNRM-CM6-1', 'CNRM-ESM2-1', 'CanESM5', 'E3SM-1-1', 'FGOALS-f3-L',
               'FGOALS-g3', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR',
               'MPI-ESM1-2-LR' , 'MRI-ESM2-0', 'NorESM2-LM', 'NorESM2-MM', 'UKESM1-0-LL']
quantiles_mesmer = ['0.05', '0.25', '0.5', '0.75', '0.95']
scenarios_mesmer = ['CurPol', 'GS', 'LD','ModAct','Ref','Ren','Neg', 'SP','ssp119','ssp534-over']
