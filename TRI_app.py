import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from stmol import showmol
import py3Dmol
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="Chemical Isomer Analysis", layout="wide")

# 2. النوت العلمية (المرجع الخاص بك)
st.markdown("""
<div style="background-color: #fdf2f2; padding: 15px; border-radius: 10px; border-left: 5px solid #800000; margin-bottom: 20px;">
    <strong style="color: #800000; font-size: 1.2em;">Stereoisomerism Reference Guide:</strong><br>
    <ul style="list-style-type: none; padding-left: 0; margin-top: 10px; color: black;">
        <li>1. <b>Cis / Trans (Relative):</b> Identical groups on same/opposite sides.</li>
        <li>2. <b>E / Z (Absolute - CIP System):</b> High-priority groups together (Z) or opposite (E).</li>
        <li>3. <b>R / S (Optical):</b> Absolute configuration of chiral centers.</li>
        <li>4. <b>Ra / Sa (Axial):</b> Stereochemistry of Allenes (C=C=C).</li>
    </ul>
    <small style="color: #555;">*Note: E/Z is required when all 4 groups on the double bond are different.</small>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 style='color: #800000; font-family: serif; border-bottom: 2px solid #dcdde1;'>Professional Isomer Analysis System</h2>", unsafe_allow_html=True)

# 3. دالة الرسم الاحترافية (تطبق قواعد النوتس الثلاثة)
def render_smart_2d(mol):
    if mol is None: return None
    
    m = Chem.Mol(mol)
    # تفعيل حسابات الكيمياء الفراغية
    Chem.AssignStereochemistry(m, force=True, cleanIt=True)
    
    is_allene = m.HasSubstructMatch(Chem.MolFromSmarts("C=C=C"))
    m = Chem.AddHs(m) if is_allene else Chem.RemoveHs(m)
    
    # محاولة توليد إحداثيات لضبط الـ Wedges
    if AllChem.EmbedMolecule(m, maxAttempts=5000, randomSeed=42) != -1:
        AllChem.Compute2DCoords(m)
        Chem.WedgeMolBonds(m, m.GetConformer())
    else:
        AllChem.Compute2DCoords(m)

    # تطبيق تسميات Cis/Trans و E/Z يدوياً بناءً على النوتس
    for bond in m.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            # القاعدة 1: Cis / Trans
            if stereo == Chem.BondStereo.STEREOCIS:
                bond.SetProp("bondNote", "Cis")
            elif stereo == Chem.BondStereo.STEREOTRANS:
                bond.SetProp("bondNote", "Trans")
            # القاعدة 2: E / Z
            elif stereo == Chem.BondStereo.STEREOE:
                bond.SetProp("bondNote", "E")
            elif stereo == Chem.BondStereo.STEREOZ:
                bond.SetProp("bondNote", "Z")

    d_opts = Draw.MolDrawOptions()
    
    # القاعدة 3 و 4: إظهار R/S و Ra/Sa فوق الذرات
    d_opts.addStereoAnnotation = True 
    
    # تفعيل إظهار نصوص الروابط (Cis/Trans/E/Z)
    d_opts.addBondLabelAnnotations = True 
    
    # إخفاء النصوص الخارجية (Legend)
    d_opts.legendFontSize = 0 
    
    if is_allene:
        d_opts.bondLineWidth = 3.0
        d_opts.minFontSize = 18
    else:
        d_opts.bondLineWidth = 1.6
        d_opts.minFontSize = 14

    img = Draw.MolToImage(m, size=(500, 500), options=d_opts, legend="")
    return img

# 4. دالة حساب Ra/Sa للألين برمجياً (للعنوان)
def get_allene_stereo(mol):
    try:
        m = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m, maxAttempts=1000) == -1: return ""
        conf = m.GetConformer()
        for b in m.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE:
                a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
                for nb in a2.GetBonds():
                    if nb.GetIdx() == b.GetIdx(): continue
                    if nb.GetBondType() == Chem.BondType.DOUBLE:
                        a3 = nb.GetOtherAtom(a2)
                        l_subs = sorted([n for n in a1.GetNeighbors() if n.GetIdx()!=a2.GetIdx()], key=lambda x: x.GetAtomicNum(), reverse=True)
                        r_subs = sorted([n for n in a3.GetNeighbors() if n.GetIdx()!=a2.GetIdx()], key=lambda x: x.GetAtomicNum(), reverse=True)
                        if l_subs and r_subs:
                            p1, p3 = np.array(conf.GetAtomPosition(a1.GetIdx())), np.array(conf.GetAtomPosition(a3.GetIdx()))
                            pl, pr = np.array(conf.GetAtomPosition(l_subs[0].GetIdx())), np.array(conf.GetAtomPosition(r_subs[0].GetIdx()))
                            dot = np.dot(np.cross(pl-p1, p3-p1), pr-p3)
                            return "Ra" if dot > 0 else "Sa"
    except: return ""
    return ""

# 5. واجهة المستخدم
name = st.text_input("Enter Structure Name (e.g., Cis-2-butene, Glucose, 2,3-pentadiene):", value="")

if st.button("Analyze & Visualize") and name:
    try:
        results = pcp.get_compounds(name, 'name')
        if results:
            base_mol = Chem.MolFromSmiles(results[0].smiles)
            pattern = Chem.MolFromSmarts("C=C=C")
            
            if base_mol.HasSubstructMatch(pattern):
                for match in base_mol.GetSubstructMatches(pattern):
                    base_mol.GetAtomWithIdx(match[0]).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

            opts = StereoEnumerationOptions(tryEmbedding=True, onlyUnassigned=False)
            isomers = list(EnumerateStereoisomers(base_mol, options=opts))
            
            if len(isomers) == 1 and base_mol.HasSubstructMatch(pattern):
                iso2 = Chem.Mol(isomers[0])
                for a in iso2.GetAtoms():
                    tag = a.GetChiralTag()
                    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                isomers.append(iso2)

            st.write("---")
            isomers_names = [] 
            cols = st.columns(len(isomers))
            
            for i, iso in enumerate(isomers):
                with cols[i]:
                    iso.ClearComputedProps()
                    Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                    
                    label = get_allene_stereo(iso)
                    if i > 0 and label in isomers_names:
                        label = "Sa" if isomers_names[0] == "Ra" else "Ra"
                    isomers_names.append(label)

                    st.markdown(f"#### Isomer {i+1}: <span style='color: #800000;'>{label}</span>", unsafe_allow_html=True)
                    st.image(render_smart_2d(iso), use_container_width=True)
                    
                    # عرض 3D
                    m3d = Chem.AddHs(iso)
                    if AllChem.EmbedMolecule(m3d, maxAttempts=2000) != -1:
                        mblock = Chem.MolToMolBlock(m3d)
                        view = py3Dmol.view(width=300, height=300)
                        view.addModel(mblock, 'mol')
                        view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
                        view.zoomTo()
                        showmol(view)
        else:
            st.error("Compound not found.")
    except Exception as e:
        st.error(f"Error: {e}")
