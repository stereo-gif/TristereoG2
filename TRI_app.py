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

# دالة ذكية للتفرقة بين Cis/Trans و E/Z حسب عدد المجموعات (تطبيق النوتس)
def get_custom_bond_label(mol):
    labels = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            # الحصول على الذرات المرتبطة بالرابطة المزدوجة
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            
            # جمع أنواع الذرات (أو المجموعات) المحيطة بالرابطة
            substituents = []
            for n in a1.GetNeighbors():
                if n.GetIdx() != a2.GetIdx(): substituents.append(n.GetSymbol())
            for n in a2.GetNeighbors():
                if n.GetIdx() != a1.GetIdx(): substituents.append(n.GetSymbol())
            
            # فحص عدد المجموعات الفريدة
            unique_subs = set(substituents)
            stereo = bond.GetStereo()
            
            # تطبيق القاعدة: لو فيه مجموعات متطابقة (Identical) -> Cis/Trans
            # لو الأربعة مختلفين (أو مفيش تكرار واضح) -> E/Z
            if len(unique_subs) < len(substituents) or len(substituents) < 4:
                if stereo == Chem.BondStereo.STEREOCIS or stereo == Chem.BondStereo.STEREOZ:
                    labels.append("Cis")
                elif stereo == Chem.BondStereo.STEREOTRANS or stereo == Chem.BondStereo.STEREOE:
                    labels.append("Trans")
            else:
                # القاعدة 2: E/Z لما الـ 4 مجموعات يكونوا مختلفين
                if stereo == Chem.BondStereo.STEREOE: labels.append("E")
                elif stereo == Chem.BondStereo.STEREOZ: labels.append("Z")
                
    return " / ".join(labels) if labels else ""

# 3. دالة الرسم (R/S تظهر داخل الرسمة فقط)
def render_smart_2d(mol):
    if mol is None: return None
    m = Chem.Mol(mol)
    is_allene = m.HasSubstructMatch(Chem.MolFromSmarts("C=C=C"))
    m = Chem.AddHs(m) if is_allene else Chem.RemoveHs(m)
    
    if AllChem.EmbedMolecule(m, maxAttempts=5000, randomSeed=42) != -1:
        AllChem.Compute2DCoords(m)
        Chem.WedgeMolBonds(m, m.GetConformer())
    else:
        AllChem.Compute2DCoords(m)

    d_opts = Draw.MolDrawOptions()
    d_opts.addStereoAnnotation = True # لظهور R/S و Ra/Sa
    d_opts.legendFontSize = 0 
    
    if is_allene:
        d_opts.bondLineWidth = 3.0
        d_opts.minFontSize = 18
    else:
        d_opts.bondLineWidth = 1.6
        d_opts.minFontSize = 14

    img = Draw.MolToImage(m, size=(500, 500), options=d_opts, legend="")
    return img

# 4. دالة حساب Ra/Sa للألين (للعنوان)
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
name = st.text_input("Enter Structure Name:", value="")

if st.button("Analyze & Visualize") and name:
    results = pcp.get_compounds(name, 'name')
    if results:
        base_mol = Chem.MolFromSmiles(results[0].smiles)
        # تفعيل حسابات الكيمياء الفراغية
        Chem.AssignStereochemistry(base_mol, force=True, cleanIt=True)
        
        pattern = Chem.MolFromSmarts("C=C=C")
        if base_mol.HasSubstructMatch(pattern):
            for match in base_mol.GetSubstructMatches(pattern):
                base_mol.GetAtomWithIdx(match[0]).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

        opts = StereoEnumerationOptions(tryEmbedding=True, onlyUnassigned=False)
        isomers = list(EnumerateStereoisomers(base_mol, options=opts))
        
        # تصحيح الألين
        if len(isomers) == 1 and base_mol.HasSubstructMatch(pattern):
            iso2 = Chem.Mol(isomers[0])
            for a in iso2.GetAtoms():
                tag = a.GetChiralTag()
                if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            isomers.append(iso2)

        st.write("---")
        cols = st.columns(len(isomers))
        
        for i, iso in enumerate(isomers):
            with cols[i]:
                iso.ClearComputedProps()
                Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                
                # فحص القواعد بالترتيب
                axial_label = get_allene_stereo(iso) 
                bond_label = get_custom_bond_label(iso)
                
                final_label = axial_label if axial_label else bond_label
                
                st.markdown(f"#### Isomer {i+1}: <span style='color: #800000;'>{final_label}</span>", unsafe_allow_html=True)
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
