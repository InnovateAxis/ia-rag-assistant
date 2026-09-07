#!/usr/bin/env python3
"""
Generate synthetic corpus for tenant isolation testing.

Produces 111 documents across three tenants (Acme, Globex, Meridian) with
deliberate overlaps to test tenant isolation in RAG retrieval:
- All three tenants have a "Fuel Surcharge Policy" with different numbers
- All three tenants have a "Detention SOP" with different free-time windows
- Acme and Globex both reference carrier "Kestrel Haulage" with different rates

All documents are synthetic and for testing only.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta

# Tenant configuration
# Total documents: Acme 42, Globex 38, Meridian 31 = 111 total
TENANTS = {
    "acme": {
        "display_name": "Acme Freight",
        "fuel_surcharge_pct": 15.5,
        "detention_free_hours": 24,
        "kestrel_rate": 2850,
        "documents": {"contracts": 8, "rate_sheets": 12, "sops": 9, "shipments": 13},
    },
    "globex": {
        "display_name": "Globex Logistics",
        "fuel_surcharge_pct": 18.2,
        "detention_free_hours": 48,
        "kestrel_rate": 3200,
        "documents": {"contracts": 7, "rate_sheets": 11, "sops": 8, "shipments": 12},
    },
    "meridian": {
        "display_name": "Meridian Transport",
        "fuel_surcharge_pct": 16.8,
        "detention_free_hours": 36,
        "documents": {"contracts": 6, "rate_sheets": 10, "sops": 7, "shipments": 8},
    },
}

CARRIERS = ["FedEx Logistics", "UPS Freight", "Kestrel Haulage", "Roadway Express"]


def generate_contract(tenant_slug, tenant_config, doc_num, is_carrier_agreement=False):
    """Generate a contract document."""
    if is_carrier_agreement:
        # Carrier agreement - reference the carrier
        if tenant_slug in ("acme", "globex"):
            carrier_name = "Kestrel Haulage"
            rate = tenant_config.get("kestrel_rate", 2800)
            return f"""CARRIER AGREEMENT - {carrier_name}

Effective Date: 2024-01-01
Parties: {tenant_config['display_name']} and {carrier_name}

SCOPE OF SERVICES
This agreement establishes the exclusive freight forwarding relationship between the parties.

PRICING
Standard LTL Rate: ${rate}/hundred weight
This rate applies to shipments of 1-100 lbs.
Minimum charge: $125 per shipment

FUEL SURCHARGE
See attached Fuel Surcharge Policy for current adjustment formula.

TERM: 24 months with annual renewal option
"""
        else:
            return f"""CARRIER AGREEMENT - Random Carriers

Effective Date: 2024-01-01
Parties: {tenant_config['display_name']} and Various Carriers

SCOPE OF SERVICES
Standard freight forwarding arrangement.

PRICING
Rates vary by carrier selection.

TERM: 12 months with auto-renewal
"""
    else:
        # MSA or amendment
        doc_type = "Master Service Agreement" if doc_num % 2 == 0 else "Amendment"
        return f"""{doc_type}

Document ID: MSA-{tenant_slug.upper()}-{doc_num:03d}
Effective Date: 2024-01-01

OVERVIEW
This document establishes service terms and conditions for freight operations.

SERVICE LEVELS
- Pickup: Within 4 business hours of booking
- Delivery: Per agreed timeline
- Customer Support: 24/7 hotline

LIABILITY
Standard freight liability per DOT regulations.

DISPUTE RESOLUTION
Disputes resolved under {tenant_config['display_name']}'s operational guidelines.
"""


def generate_contract_filename(tenant_slug: str, is_carrier: bool, doc_num: int) -> str:
    """Generate contract filename.

    Only Acme and Globex have Kestrel carrier agreements.
    Meridian has generic carrier agreements.
    """
    if is_carrier:
        if tenant_slug in ("acme", "globex"):
            return f"carrier_agreement_kestrel_haulage_{doc_num:02d}.txt"
        else:  # meridian
            return f"carrier_agreement_generic_{doc_num:02d}.txt"
    else:
        return f"msa_amendment_{doc_num:02d}.txt"


def generate_rate_sheet(tenant_slug, tenant_config, doc_num):
    """Generate a rate sheet document."""
    sheet_types = [
        "Zone Pricing",
        "Fuel Surcharge Policy",
        "Accessorial Charges",
        "Heavy Specialized Equipment",
        "International Rates",
    ]

    sheet_type = sheet_types[doc_num % len(sheet_types)]

    if "Fuel Surcharge" in sheet_type:
        # The deliberate overlap - all three tenants have this with different numbers
        return f"""FUEL SURCHARGE POLICY

Effective: 2024-01-01
Tenant: {tenant_config['display_name']}

BASE FUEL SURCHARGE: {tenant_config['fuel_surcharge_pct']}%

This surcharge applies to all line-haul charges and is adjusted monthly based on
the national average diesel index.

CALCULATION
Base freight charge × {tenant_config['fuel_surcharge_pct']}% = Fuel surcharge

ADJUSTMENT
The surcharge percentage changes on the first of each month based on prevailing
market conditions as reported by the DOT Energy Information Administration.

CUSTOMER NOTIFICATION
Customers receive 24 hours notice of surcharge changes.
"""
    else:
        return f"""{sheet_type}

Rate Sheet: RS-{tenant_slug.upper()}-{doc_num:03d}
Published: 2024-09-01
Tenant: {tenant_config['display_name']}

{sheet_type} RATES

Zone A (0-100 miles):
  LTL: $0.85/lb minimum $85
  Full: $1200 per load

Zone B (100-300 miles):
  LTL: $0.95/lb minimum $120
  Full: $1600 per load

Zone C (300+ miles):
  LTL: $1.10/lb minimum $180
  Full: $2200 per load

Accessorial charges apply for hazmat, oversize, and time-definite services.
"""


def generate_sop(tenant_slug, tenant_config, doc_num):
    """Generate a Standard Operating Procedure document."""
    sop_types = [
        ("Claims Processing", "claims"),
        ("Detention Procedures", "detention"),
        ("Hazmat Handling", "hazmat"),
        ("Temperature Control", "temperature"),
        ("Customs Documentation", "customs"),
    ]

    sop_name, sop_key = sop_types[doc_num % len(sop_types)]

    if "Detention" in sop_name:
        # The deliberate overlap - all three tenants have this with different free windows
        free_hours = tenant_config['detention_free_hours']
        tier2_ceiling = free_hours + 24  # Dynamic second tier: free_hours to free_hours+24
        return f"""STANDARD OPERATING PROCEDURE: {sop_name}

SOP ID: SOP-{tenant_slug.upper()}-{doc_num:03d}
Revision: 2
Last Updated: 2024-08-15
Tenant: {tenant_config['display_name']}

DETENTION TIME WINDOWS

Free Time: {free_hours} hours
This free time begins when the driver arrives at the facility.

Detention Rates (after free time):
- Hours 0-{free_hours}: No charge
- Hours {free_hours}-{tier2_ceiling}: $50 per 8-hour period
- Hours {tier2_ceiling}+: $100 per 8-hour period

PROCEDURES
1. Driver initiates detention clock upon arrival
2. Warehouse confirms in system
3. Detention charges accrue automatically after free window
4. Weekly billing to customer account

COMMUNICATION
Customer must be notified if detention will exceed 24 hours.
"""
    else:
        return f"""STANDARD OPERATING PROCEDURE: {sop_name}

SOP ID: SOP-{tenant_slug.upper()}-{doc_num:03d}
Revision: 1
Last Updated: 2024-08-01
Tenant: {tenant_config['display_name']}

OVERVIEW
This procedure governs {sop_name.lower()} for all {tenant_config['display_name']} shipments.

SCOPE
Applies to all domestic operations and contracted third-party carriers.

REQUIREMENTS
- Staff must complete annual training
- All procedures documented in writing
- Exceptions require management approval
- Customer communication within 24 hours

COMPLIANCE
Adherence to all DOT and state regulations is mandatory.
"""


def generate_shipment(tenant_slug, tenant_config, doc_num):
    """Generate a shipment/delivery report document."""
    dates = [
        datetime(2024, 8, 15) + timedelta(days=i) for i in range(30)
    ]
    date = dates[doc_num % len(dates)]

    exceptions = [
        "None",
        "Delayed 2 hours due to traffic",
        "Signature refused - rescheduled",
        "Weather delay",
        "Vehicle breakdown - covered by insurance",
    ]

    exception = exceptions[doc_num % len(exceptions)]

    return f"""DELIVERY REPORT

Shipment ID: SHP-{tenant_slug.upper()}-{date.strftime('%Y%m%d')}-{doc_num:04d}
Tenant: {tenant_config['display_name']}
Delivery Date: {date.strftime('%Y-%m-%d')}

ORIGIN
Location: {tenant_config['display_name']} Distribution Center
Address: Regional Hub, USA

DESTINATION
Recipient: Customer {doc_num}
Address: Various

FREIGHT DETAILS
Weight: {300 + (doc_num * 50) % 2000} lbs
Pieces: {1 + (doc_num % 5)}
Classification: General Freight

CARRIER: {CARRIERS[doc_num % len(CARRIERS)]}
Driver: {chr(65 + (doc_num % 26))}{chr(65 + ((doc_num + 1) % 26))} {100 + doc_num}

DELIVERY NOTES
Delivered as scheduled.
Exception: {exception}

CHARGES
Base Freight: ${800 + (doc_num * 10)}
Fuel Surcharge (see policy): ${(800 + (doc_num * 10)) * tenant_config.get('fuel_surcharge_pct', 15) / 100:.2f}
Accessorials: $0
Total: ${800 + (doc_num * 10) + ((800 + (doc_num * 10)) * tenant_config.get('fuel_surcharge_pct', 15) / 100):.2f}
"""


def generate_corpus(output_dir: Path) -> dict:
    """Generate complete corpus structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for tenant_slug, config in TENANTS.items():
        stats[tenant_slug] = {}
        tenant_dir = output_dir / tenant_slug

        # Create category directories
        for category in config["documents"].keys():
            category_dir = tenant_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)

            count = config["documents"][category]
            stats[tenant_slug][category] = count

            # Generate documents in this category
            for doc_num in range(1, count + 1):
                if category == "contracts":
                    # Mix of MSAs/amendments and carrier agreements
                    is_carrier = doc_num in (3, 4, 5)  # Documents 3-5 are carrier agreements
                    content = generate_contract(
                        tenant_slug, config, doc_num, is_carrier_agreement=is_carrier
                    )
                    doc_name = generate_contract_filename(tenant_slug, is_carrier, doc_num)
                elif category == "rate_sheets":
                    content = generate_rate_sheet(tenant_slug, config, doc_num)
                    doc_name = f"rate_sheet_{doc_num:02d}.txt"
                elif category == "sops":
                    content = generate_sop(tenant_slug, config, doc_num)
                    doc_name = f"sop_{doc_num:02d}.txt"
                else:  # shipments
                    content = generate_shipment(tenant_slug, config, doc_num)
                    doc_name = f"shipment_{doc_num:04d}.txt"

                doc_path = category_dir / doc_name
                doc_path.write_text(content, encoding="utf-8", newline="\n")

    return stats


def create_readme(output_dir: Path):
    """Create README documenting the synthetic corpus."""
    readme_content = """# Corpus — Synthetic Test Data for Tenant Isolation Testing

This directory contains synthetic documents generated for testing tenant isolation in the RAG system. **All data is completely synthetic and for testing purposes only.** No real customer documents or actual business data is included.

## Structure

```
corpus/
├── acme/          (Acme Freight, 42 documents)
├── globex/        (Globex Logistics, 38 documents)
└── meridian/      (Meridian Transport, 31 documents)
```

Each tenant has four document categories:
- `contracts/`: Master Service Agreements and carrier agreements (8 documents)
- `rate_sheets/`: Pricing, fuel surcharge policies, accessorials (12 documents)
- `sops/`: Standard Operating Procedures for claims, detention, hazmat, temperature control (9 documents)
- `shipments/`: Delivery reports with exception narratives (13 documents)

Total: 111 documents

## Deliberate Overlap for Isolation Testing

The corpus includes intentional overlaps to test that tenant isolation is real and not accidental:

1. **Fuel Surcharge Policy** — All three tenants have this document with DIFFERENT numbers:
   - Acme: 15.5%
   - Globex: 18.2%
   - Meridian: 16.8%

2. **Detention SOP** — All three tenants have this document with DIFFERENT free-time windows:
   - Acme: 24 hours free time
   - Globex: 48 hours free time
   - Meridian: 36 hours free time

3. **Kestrel Haulage Carrier Agreements** — Acme and Globex both reference the same carrier with DIFFERENT rates:
   - Acme: $2,850/cwt
   - Globex: $3,200/cwt

If a query for "Fuel Surcharge Policy" returns documents from the wrong tenant, or if a "Detention SOP" retrieves the wrong free-time window, isolation has failed.

## Generation

This corpus was generated by `generate_corpus.py` using deterministic logic. To regenerate identical documents, run:

```bash
python scripts/generate_corpus.py
```

This reproducibility ensures the corpus is version-controlled and auditable, rather than requiring 111 manual file creations.

## Synthetic Nature

Every document is generated with placeholder business logic and fictitious shipment dates and customer IDs. Common attributes:

- Tenant names and company identities are test personas
- Dates are within August-September 2024
- Shipment IDs, customer IDs, and financial figures are illustrative
- No actual carrier names correspond to real contracted services (except "Kestrel Haulage," which is invented)
- No real customer or shipment data is present
- All regulatory references (DOT, hazmat, etc.) are structural only

This corpus is safe for source control and may be freely shared as it contains no production data.
"""

    (output_dir / "README.md").write_text(readme_content, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    import sys

    # Use script location to find corpus directory
    # When run from repo root: python scripts/generate_corpus.py
    if len(sys.argv) > 1:
        corpus_dir = Path(sys.argv[1])
    else:
        corpus_dir = Path("corpus")

    print(f"Generating corpus in {corpus_dir}")
    stats = generate_corpus(corpus_dir)
    create_readme(corpus_dir)

    # Print statistics
    total = 0
    for tenant, categories in stats.items():
        tenant_total = sum(categories.values())
        total += tenant_total
        print(f"{tenant}: {tenant_total} documents")
        for category, count in categories.items():
            print(f"  {category}: {count}")

    print(f"\nTotal: {total} documents")
    print(f"README created at {corpus_dir}/README.md")
