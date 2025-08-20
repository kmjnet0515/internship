import cv2
import numpy as np



classes = ['Anchor', 'SeaReaf', 'SeaSquirt', 'WaterDrop', 'WhirlPool']

for i in range(40):
    names = classes[i//8]
    # 1. 이미지 불러오기 (흑백으로)
    img = cv2.imread(f"./DataSet2/train/images/{names}_{i%8}.jpg", cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    clone = img.copy()
    dst_points = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])
    src_points = np.float32([
        [30, 135],
        [770, 0],
        [890, 685],
        [360, 475]
    ])
    # ==============================
    # 5️⃣ 변환 행렬과 보정
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(img, matrix, (w, h))

    # ==============================
    # 6️⃣ 결과 보기
    #cv2.imshow("Warped (Front View)", result)
    #cv2.waitKey(10)
    #cv2.destroyAllWindows()

    # ==============================
    # 7️⃣ 결과 저장
    # 2. 히스토그램 평활화
    equalized = cv2.equalizeHist(img)

    # 3. 밝게/어둡게 할 gamma 값 정의
    bright_gammas = [0.6, 0.7, 0.8, 0.9]  # 밝게 (gamma < 1)
    dark_gammas   = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]       # 어둡게 (gamma > 1)

    def adjust_gamma(image, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255
                        for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)

    # 4. 밝게 변형된 이미지들 생성
    for idx, g in enumerate(bright_gammas):
        bright_img = adjust_gamma(equalized, g)
        cv2.imwrite(f"./DataSet2/train/images/{names}_{i%8}{idx+1}.jpg", bright_img)

    # 5. 어둡게 변형된 이미지들 생성
    for idx, g in enumerate(dark_gammas):
        dark_img = adjust_gamma(equalized, g)
        cv2.imwrite(f"./DataSet2/train/images/{names}_{i%8}{idx+5}.jpg", dark_img)

    # 6. 평활화된 원본 저장
    cv2.imwrite(f"./DataSet2/train/images/{names}_{i%8}{11}.jpg", equalized)

    print("데이터 증강 이미지 생성 완료!")
    input_file = f"./DataSet2/train/labels/{names}_{i%8}.txt"
    for index in range(1,12):
    # 새로 저장할 파일 이름 (직접 지정)
        new_file_name = f"./DataSet2/train/labels/{names}_{i%8}{index}.txt"
        # 파일 읽기
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 새 이름으로 저장
        with open(new_file_name, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"파일을 '{new_file_name}' 이름으로 저장했습니다!")
    