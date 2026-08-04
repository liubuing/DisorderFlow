import re

native_h3 = 'VRYDHYSGSSDY'
print(f"Native CDR-H3: {native_h3}")
print()

with open('idp_design_results/4hix_mpnn_h3/seqs/4hix_clean.fa') as f:
    for line in f:
        if line.startswith('>T=0.5'):
            parts = line.strip().split(',')
            sample = parts[1].split('=')[1].strip()
            score = float(parts[2].split('=')[1])
            recovery = float(parts[4].split('=')[1])
            seq = next(f).strip()
            chains = seq.split('/')
            h_chain = chains[1]
            m = re.search(r'YCV(.+?)WGQ', h_chain)
            if m:
                h3 = m.group(1)
                print(f"  sample {sample:>2}: score={score:.3f} rec={recovery:.2f} {h3}")
