import streamlit as st
from api import get_category

def main():
    st.title("E-Ticaret Yorum Kategorizasyonu")

    # Kullanıcıdan metin girişi al
    user_input = st.text_area("Lütfen bir yorum girin:", "")

    if st.button("Yorumu Sınıflandır"):
        result = get_category(user_input)

        if "error" in result:
            st.warning(result["error"])
        else:
            st.success(f"Yorumunuz {result['category']} kategorisine aittir.")

if __name__ == "__main__":
    main()
