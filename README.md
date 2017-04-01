# Feature_selection
use random forest to find important feature.

1. feature importance ranking:
 * ## By Feature importance ranking top10
 + 1. TB_9a
 + 2. TB_a9
 + 3. Img0.1
 + 4. ent_p_5
 + 5. TB_b1
 + 6. TB_71
 + 7. ent_p_8
 + 8. TB_ce
 + 9. GetStringTypeA
 + 10. ExitProcess

2. Useless feature ranking:
 * ## Featurn ranking list Top10 useless feature:
 + GetLastActivePopup
 + ImageList_Add
 + GlobalDeleteAtom
 + IsBadReadPtr
 + SelectPalette
 + GetMenuState
 + ExitThread
 + AdjustWindowRectEx
 + GetEnvironmentVariableA
 + SHGetFileInfoA

3. 使用 sklearn 裡面的 RandonforestClassifier，取前10為 important features，後10為useless features。
   但 useless features 只代表**相對不佳**，不代表這些 useless features 對分類**完全沒用**，important features 亦同。

4. sklearn package 中的 RandomForestClassifier 做 feature selection，而 pandas，numpy 處理資料，matplotlib.pyplot 畫圖

5. 無
