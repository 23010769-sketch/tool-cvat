"""Ánh xạ nhãn có kiểm soát; không suy đoán lớp lane/drivable/segmentation."""

import re
import unicodedata


def normalize(name):
    name = unicodedata.normalize("NFKC", name).casefold().strip()
    return re.sub(r"[\s_-]+", " ", name)


def literal_name(name):
    """Keep separators: detection 'traffic sign' differs from semantic 'traffic_sign'."""
    return unicodedata.normalize("NFKC", name).casefold().strip()


def namespaced_preferences(name):
    group, separator, base = literal_name(name).partition("/")
    if not separator or group not in ("det", "semantic", "reference"):
        return ()
    if group == "det":
        return {"person": ("pedestrian", "person"),
                "motor": ("motorcycle", "motorbike", "motor"),
                "bike": ("bicycle", "bike"),
                "traffic light": ("traffic light", "traffic_light"),
                "traffic sign": ("traffic sign", "traffic_sign")}.get(base, (base,))
    if group == "semantic" and base in ("traffic light", "traffic sign"):
        return (base.replace(" ", "_"), base)
    return (base,)


ALIASES = {
    "person": {"person", "pedestrian"},
    "car": {"car"}, "truck": {"truck"}, "bus": {"bus"},
    "bicycle": {"bicycle", "bike"}, "motorcycle": {"motorcycle", "motorbike"},
    "motor": {"motorcycle", "motorbike"}, "bike": {"bicycle", "bike"},
}


class AmbiguousMappingError(ValueError):
    def __init__(self, source, matches):
        self.source = source
        self.matches = sorted(matches, key=lambda label: label["id"])
        choices = "; ".join(f"ID {label['id']}: {label['name']!r}" for label in self.matches)
        super().__init__(f"Mapping lớp {source} khớp nhiều label: {choices}. "
                         "Chọn ID trong Terminal hoặc đặt tên_lớp = ID dưới [labels] trong annotation.toml.")


def label_candidates(names, labels, overrides=None, *, shape_types=None):
    """Resolve possible targets without choosing between ambiguous labels."""
    literal_overrides = {normalize(k): literal_name(v) for k, v in (overrides or {}).items() if isinstance(v, str)}
    overrides = {normalize(k): normalize(v) if isinstance(v, str) else v
                 for k, v in (overrides or {}).items()}
    shape_types = shape_types or {key: "rectangle" for key in names}
    allowed_types = set(shape_types.values()) | {"any"}
    targets = {}
    literal_targets = {}
    for label in labels:
        if label.get("type", "any") in allowed_types and not label.get("parent"):
            targets.setdefault(normalize(label["name"]), []).append(label)
            literal_targets.setdefault(literal_name(label["name"]), []).append(label)
    by_id = {label["id"]: label for matches in targets.values() for label in matches}
    source_names = {normalize(name) for name in names.values()}
    if set(overrides) - source_names:
        raise ValueError("Mapping tùy chỉnh chứa tên lớp không tồn tại trong model.")
    for key, value in overrides.items():
        if value == "":
            continue  # An explicit empty mapping skips this class.
        valid = (value in by_id if type(value) is int else isinstance(value, str) and value in targets)
        if not valid:
            raise ValueError(f"Mapping {key}: label đích không có hoặc không cho phép rectangle.")
    result = {}
    for class_id, name in names.items():
        source = normalize(name)
        if source in overrides and overrides[source] == "":
            result[class_id] = []
            continue
        if source in overrides and type(overrides[source]) is int:
            matches = [by_id[overrides[source]]]
        elif source in overrides and literal_overrides.get(source) in literal_targets:
            matches = literal_targets[literal_overrides[source]]
        else:
            matches = []
            if source not in overrides:
                preferences = namespaced_preferences(name)
                # First use the actual spelling in the live Job, then aliases.
                # Missing classes remain unmapped instead of requiring old Job IDs.
                for candidate in preferences:
                    if candidate in literal_targets:
                        matches = literal_targets[candidate]
                        break
                if not matches:
                    for candidate in preferences:
                        if normalize(candidate) in targets:
                            matches = targets[normalize(candidate)]
                            break
            if not matches:
                candidates = {overrides[source]} if source in overrides else ALIASES.get(source, {source})
                matches = [label for candidate in candidates for label in targets.get(candidate, [])]
        # Repeated records with the same ID do not represent different labels.
        matches = list({label["id"]: label for label in matches}.values())
        required_type = shape_types.get(class_id, "rectangle")
        result[class_id] = [label for label in matches if label.get("type", "any") in ("any", required_type)]
        if matches and not result[class_id] and source in overrides:
            raise ValueError(f"Mapping {name}: label không cho phép geometry {required_type}.")
    # A Job with only one person/sign/light label should receive the detector's
    # boxes, rather than two competing sources for the same target. Explicit
    # mappings remain authoritative, including older Job 206 profiles.
    detection_ids = {matches[0]["id"] for key, matches in result.items()
                     if literal_name(names[key]).startswith("det/") and len(matches) == 1}
    for key, matches in result.items():
        if (normalize(names[key]) not in overrides and literal_name(names[key]).startswith("semantic/")
                and len(matches) == 1 and matches[0]["id"] in detection_ids):
            result[key] = []
    return result


class LabelMapper:
    def __init__(self, names, labels, overrides=None, *, shape_types=None):
        self.mapping = {}
        self.unmapped = []
        for class_id, matches in label_candidates(names, labels, overrides, shape_types=shape_types).items():
            if len(matches) > 1:
                raise AmbiguousMappingError(names[class_id], matches)
            if matches:
                self.mapping[class_id] = matches[0]
            else:
                self.unmapped.append(names[class_id])
        if not self.mapping:
            raise ValueError("Không có label rectangle nào ánh xạ được với model; kiểm tra labels/mapping.")

    def label(self, class_id):
        return self.mapping.get(class_id)

    @staticmethod
    def attributes(label):
        result = []
        for attribute in label.get("attributes", []):
            if "default_value" not in attribute or attribute["default_value"] is None:
                raise ValueError("Label có thuộc tính thiếu default_value; cần cấu hình trước khi ghi.")
            result.append({"spec_id": attribute["id"], "value": str(attribute["default_value"])})
        return result
