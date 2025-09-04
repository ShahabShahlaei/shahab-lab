from common_imports import *
def diagnose_model_inputs(model_data):
    for model_name, model_dict in model_data.items():
        print(f"\n=== Model: {model_name} ===")
        try:
            # Check both experiments
            for experiment in ['2xDMS', 'control']:
                if experiment not in model_dict:
                    print(f"  ⚠ Experiment '{experiment}' missing.")
                    continue

                # Check variables
                for var in ['rlut', 'rsut', 'rsdt']:
                    if var not in model_dict[experiment]:
                        print(f"  ⚠ Variable '{var}' missing in {experiment}.")
                        continue

                    da = model_dict[experiment][var]
                    print(f"  ✅ {var} - {experiment}")
                    print(f"    Time range: {da.time.values[0]} to {da.time.values[-1]}  ({len(da.time)} steps)")
                    print(f"    Lat: {da.sizes.get('lat', 'NA')}  Lon: {da.sizes.get('lon', 'NA')}")

        except Exception as e:
            print(f"  ❌ Error diagnosing {model_name} - {experiment} - {var}: {e}")

        # Check time overlap
        try:
            t2xdms = model_dict['2xDMS']['rlut'].time
            tcontrol = model_dict['control']['rlut'].time
            overlap = np.intersect1d(t2xdms.values, tcontrol.values)
            if len(overlap) == 0:
                print(f"  ❗ No overlapping time between '2xDMS' and 'control' runs.")
            else:
                print(f"  ✅ Overlap length: {len(overlap)} steps ({overlap[0]} to {overlap[-1]})")
        except Exception as e:
            print(f"  ❌ Could not check time overlap: {e}")

        # Check grid compatibility (lat/lon)
        try:
            lat_2xdms = model_dict['2xDMS']['rlut'].lat
            lat_control = model_dict['control']['rlut'].lat
            if lat_2xdms.equals(lat_control):
                print(f"  ✅ Lat grids match.")
            else:
                print(f"  ❗ Lat grids mismatch: {lat_2xdms.shape} vs {lat_control.shape}")

            lon_2xdms = model_dict['2xDMS']['rlut'].lon
            lon_control = model_dict['control']['rlut'].lon
            if lon_2xdms.equals(lon_control):
                print(f"  ✅ Lon grids match.")
            else:
                print(f"  ❗ Lon grids mismatch: {lon_2xdms.shape} vs {lon_control.shape}")
        except Exception as e:
            print(f"  ❌ Could not check lat/lon grids: {e}")
