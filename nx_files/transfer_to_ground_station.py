from pathlib import Path
import argparse
import csv
import os
import tempfile
import time
import zipfile
import requests

from infer_scenes import TRTInference, infer_scene


SCENES_ROOT = Path("/mnt/usb/scenes")
SCRIPT_DIR = Path(__file__).resolve().parent
CERT_PATH = SCRIPT_DIR / "certs" / "rootCA.pem"

GROUND_STATION_SCENES_URL = "https://192.168.1.31:8443/upload_scene"
GROUND_STATION_METRICS_URL = "https://192.168.1.31:8443/upload_metrics"
GROUND_STATION_CSV_URL = "https://192.168.1.31:8443/upload_nx_csv"

UNET_ENGINE = SCRIPT_DIR / "Unet_fp16.engine"
UNETPP_ENGINE = SCRIPT_DIR / "UnetPlusPlus_fp16.engine"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer scenes to ground station over HTTPS"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "unet", "unetplusplus"],
        required=True,
        help="Transfer mode",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Cloud fraction threshold for model-filtered modes",
    )
    return parser.parse_args()


def make_session_id(mode: str) -> str:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{mode}"


def zip_scene(scene_dir: Path):
    tmp_dir = Path(tempfile.mkdtemp(prefix="scene_zip_"))
    zip_path = tmp_dir / f"{scene_dir.name}.zip"

    start = time.time()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(scene_dir.iterdir()):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
    zip_time_sec = time.time() - start

    return zip_path, zip_time_sec


def upload_scene(
    scene_dir: Path,
    session_id: str,
    mode: str,
    cloud_fraction,
    threshold,
    inference_time_sec,
):
    zip_path, zip_time_sec = zip_scene(scene_dir)
    zip_bytes = zip_path.stat().st_size

    upload_start = time.time()
    with open(zip_path, "rb") as f:
        files = {"file": (zip_path.name, f, "application/zip")}
        data = {
            "scene_id": scene_dir.name,
            "mode": mode,
            "session_id": session_id,
            "decision": "send",
            "cloud_fraction": str(cloud_fraction),
            "threshold": str(threshold),
            "sender_inference_time_sec": f"{inference_time_sec:.6f}",
            "sender_zip_time_sec": f"{zip_time_sec:.6f}",
            "sender_upload_time_sec": "0.0",
            "sender_end_to_end_time_sec": "0.0",
        }

        response = requests.post(
            GROUND_STATION_SCENES_URL,
            files=files,
            data=data,
            verify=str(CERT_PATH),
            timeout=600,
        )

    upload_time_sec = time.time() - upload_start
    end_to_end_time_sec = inference_time_sec + zip_time_sec + upload_time_sec

    try:
        os.remove(zip_path)
        zip_path.parent.rmdir()
    except Exception:
        pass

    return {
        "zip_bytes": zip_bytes,
        "zip_time_sec": zip_time_sec,
        "upload_time_sec": upload_time_sec,
        "end_to_end_time_sec": end_to_end_time_sec,
        "status_code": response.status_code,
        "response_text": response.text,
    }


def send_metrics(
    session_id: str,
    scene_id: str,
    mode: str,
    decision: str,
    cloud_fraction,
    threshold,
    inference_time_sec,
    zip_time_sec,
    upload_time_sec,
    end_to_end_time_sec,
    status_code,
):
    data = {
        "session_id": session_id,
        "scene_id": scene_id,
        "mode": mode,
        "decision": decision,
        "cloud_fraction": str(cloud_fraction),
        "threshold": str(threshold),
        "sender_inference_time_sec": f"{inference_time_sec:.6f}",
        "sender_zip_time_sec": f"{zip_time_sec:.6f}",
        "sender_upload_time_sec": f"{upload_time_sec:.6f}",
        "sender_end_to_end_time_sec": f"{end_to_end_time_sec:.6f}",
        "status_code": str(status_code),
    }

    try:
        response = requests.post(
            GROUND_STATION_METRICS_URL,
            data=data,
            verify=str(CERT_PATH),
            timeout=60,
        )

        if response.status_code != 200:
            print(
                f"Warning: metrics upload failed for {scene_id} "
                f"with status {response.status_code}: {response.text}"
            )

        return response

    except requests.RequestException as e:
        print(f"Warning: metrics upload error for {scene_id}: {e}")
        return None


def upload_csv_file(session_id: str, csv_path: Path, csv_type: str):
    with open(csv_path, "rb") as f:
        files = {"file": (csv_path.name, f, "text/csv")}
        data = {
            "session_id": session_id,
            "csv_type": csv_type,
        }

        response = requests.post(
            GROUND_STATION_CSV_URL,
            files=files,
            data=data,
            verify=str(CERT_PATH),
            timeout=120,
        )

    return response


def choose_model(mode: str):
    if mode == "all":
        return None, "all"

    if mode == "unet":
        if not UNET_ENGINE.exists():
            raise FileNotFoundError(f"U-Net engine not found: {UNET_ENGINE}")
        return TRTInference(str(UNET_ENGINE)), "Unet_fp16"

    if mode == "unetplusplus":
        if not UNETPP_ENGINE.exists():
            raise FileNotFoundError(f"U-Net++ engine not found: {UNETPP_ENGINE}")
        return TRTInference(str(UNETPP_ENGINE)), "UnetPlusPlus_fp16"

    raise ValueError(f"Invalid mode: {mode}")


def main():
    session_start = time.time()
    session_start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    args = parse_args()
    mode = args.mode
    cloud_threshold = args.threshold

    if not SCENES_ROOT.exists():
        raise FileNotFoundError(f"Scenes root not found: {SCENES_ROOT}")
    if not CERT_PATH.exists():
        raise FileNotFoundError(f"Certificate file not found: {CERT_PATH}")

    session_id = make_session_id(mode)
    csv_path = SCRIPT_DIR / f"{session_id}_results.csv"
    summary_csv_path = SCRIPT_DIR / f"{session_id}_session_summary.csv"

    scene_dirs = sorted([p for p in SCENES_ROOT.iterdir() if p.is_dir()])
    if not scene_dirs:
        raise ValueError(f"No scene folders found in: {SCENES_ROOT}")

    trt_model, model_name = choose_model(mode)

    rows = []

    for i, scene_dir in enumerate(scene_dirs, start=1):
        print(f"\n=== [{i}/{len(scene_dirs)}] {scene_dir.name} ===")

        try:
            if mode == "all":
                cloud_fraction = 0.0
                inference_time = 0.0
                decision = "send"
            else:
                result = infer_scene(scene_dir, trt_model)
                cloud_fraction = result["cloud_fraction"]
                inference_time = result["elapsed_sec"]
                decision = "send" if cloud_fraction < cloud_threshold else "discard"

            row = {
                "session_id": session_id,
                "mode": mode,
                "scene_id": scene_dir.name,
                "cloud_fraction": cloud_fraction,
                "threshold": cloud_threshold,
                "decision": decision,
                "inference_time_sec": inference_time,
                "zip_bytes": "",
                "zip_time_sec": "",
                "upload_time_sec": "",
                "end_to_end_time_sec": "",
                "status_code": "",
                "error": "",
            }

            if decision == "send":
                upload_result = upload_scene(
                    scene_dir=scene_dir,
                    session_id=session_id,
                    mode=mode,
                    cloud_fraction=cloud_fraction,
                    threshold=cloud_threshold,
                    inference_time_sec=inference_time,
                )

                row["zip_bytes"] = upload_result["zip_bytes"]
                row["zip_time_sec"] = upload_result["zip_time_sec"]
                row["upload_time_sec"] = upload_result["upload_time_sec"]
                row["end_to_end_time_sec"] = upload_result["end_to_end_time_sec"]
                row["status_code"] = upload_result["status_code"]

                send_metrics(
                    session_id=session_id,
                    scene_id=scene_dir.name,
                    mode=mode,
                    decision="send",
                    cloud_fraction=cloud_fraction,
                    threshold=cloud_threshold,
                    inference_time_sec=inference_time,
                    zip_time_sec=upload_result["zip_time_sec"],
                    upload_time_sec=upload_result["upload_time_sec"],
                    end_to_end_time_sec=upload_result["end_to_end_time_sec"],
                    status_code=upload_result["status_code"],
                )

            else:
                send_metrics(
                    session_id=session_id,
                    scene_id=scene_dir.name,
                    mode=mode,
                    decision="discard",
                    cloud_fraction=cloud_fraction,
                    threshold=cloud_threshold,
                    inference_time_sec=inference_time,
                    zip_time_sec=0.0,
                    upload_time_sec=0.0,
                    end_to_end_time_sec=inference_time,
                    status_code=0,
                )

            rows.append(row)

        except Exception as e:
            print(f"Failed on {scene_dir.name}: {e}")
            rows.append(
                {
                    "session_id": session_id,
                    "mode": mode,
                    "scene_id": scene_dir.name,
                    "cloud_fraction": "",
                    "threshold": cloud_threshold,
                    "decision": "error",
                    "inference_time_sec": "",
                    "zip_bytes": "",
                    "zip_time_sec": "",
                    "upload_time_sec": "",
                    "end_to_end_time_sec": "",
                    "status_code": "",
                    "error": str(e),
                }
            )

    fieldnames = [
        "session_id",
        "mode",
        "scene_id",
        "cloud_fraction",
        "threshold",
        "decision",
        "inference_time_sec",
        "zip_bytes",
        "zip_time_sec",
        "upload_time_sec",
        "end_to_end_time_sec",
        "status_code",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved recomputer per-scene results CSV: {csv_path}")

    session_end = time.time()
    session_end_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    session_elapsed_sec = session_end - session_start

    num_scenes_total = len(rows)
    num_scenes_sent = sum(1 for r in rows if r["decision"] == "send")
    num_scenes_discarded = sum(1 for r in rows if r["decision"] == "discard")
    num_scene_errors = sum(1 for r in rows if r["decision"] == "error")

    total_inference_time_sec = sum(
        float(r["inference_time_sec"]) for r in rows if r["inference_time_sec"] != ""
    )
    total_zip_bytes = sum(int(r["zip_bytes"]) for r in rows if r["zip_bytes"] != "")
    total_zip_time_sec = sum(
        float(r["zip_time_sec"]) for r in rows if r["zip_time_sec"] != ""
    )
    total_upload_time_sec = sum(
        float(r["upload_time_sec"]) for r in rows if r["upload_time_sec"] != ""
    )
    total_end_to_end_time_sec = sum(
        float(r["end_to_end_time_sec"]) for r in rows if r["end_to_end_time_sec"] != ""
    )

    summary_fieldnames = [
        "session_id",
        "mode",
        "threshold",
        "session_start_utc",
        "session_end_utc",
        "session_elapsed_sec",
        "num_scenes_total",
        "num_scenes_sent",
        "num_scenes_discarded",
        "num_scene_errors",
        "total_inference_time_sec",
        "total_zip_bytes",
        "total_zip_time_sec",
        "total_upload_time_sec",
        "total_end_to_end_time_sec",
    ]

    summary_row = {
        "session_id": session_id,
        "mode": mode,
        "threshold": cloud_threshold,
        "session_start_utc": session_start_utc,
        "session_end_utc": session_end_utc,
        "session_elapsed_sec": session_elapsed_sec,
        "num_scenes_total": num_scenes_total,
        "num_scenes_sent": num_scenes_sent,
        "num_scenes_discarded": num_scenes_discarded,
        "num_scene_errors": num_scene_errors,
        "total_inference_time_sec": total_inference_time_sec,
        "total_zip_bytes": total_zip_bytes,
        "total_zip_time_sec": total_zip_time_sec,
        "total_upload_time_sec": total_upload_time_sec,
        "total_end_to_end_time_sec": total_end_to_end_time_sec,
    }

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Saved recomputer session summary CSV: {summary_csv_path}")

    print("\nUploading recomputer CSV files to ground station...")

    results_response = upload_csv_file(
        session_id=session_id,
        csv_path=csv_path,
        csv_type="results",
    )
    print(
        f"Uploaded results CSV: status={results_response.status_code}, "
        f"response={results_response.text}"
    )

    summary_response = upload_csv_file(
        session_id=session_id,
        csv_path=summary_csv_path,
        csv_type="session_summary",
    )
    print(
        f"Uploaded summary CSV: status={summary_response.status_code}, "
        f"response={summary_response.text}"
    )


if __name__ == "__main__":
    main()
